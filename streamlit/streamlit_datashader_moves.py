import streamlit as st
import pandas as pd
import numpy as np
from geolib import geohash as geolib
import holoviews as hv
import hvplot.pandas  # noqa
from holoviews.operation.datashader import datashade
import datashader as ds
from datashader.utils import lnglat_to_meters
from streamlit_bokeh import streamlit_bokeh

import colorcet
import os

hv.extension('bokeh')

# ------------------- SETUP -------------------
st.set_page_config(layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, sep=";")

    # Clean timestamps
    df['start_time'] = df['start_time'].str[:-6]
    df['end_time'] = df['end_time'].str[:-6]
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['distance(m)'] = df['distance(m)'].astype(int)
    df['gCO2'] = df['gCO2'].astype(int)

    df = df.groupby(['participant_id', 'start_geohash', 'start_time'])['distance(m)'].max().reset_index().merge(
        df, on=['participant_id', 'start_geohash', 'start_time', 'distance(m)']
    )
    return df


def geohash_to_coordinate(geohash):
    try:
        lat, lon = geolib.decode(geohash)
        return np.array([float(lat), float(lon)])
    except:
        return np.array([0.0, 0.0])


def translate_mot(mots):
    res = set()
    for mot in mots:
        if mot == "Car":
            res.update(["CAR", "ELECTRIC_CAR", "HYBRID_CAR"])
        elif mot == "Train":
            res.update(["TRAIN"])
        elif mot == "Walking":
            res.update(["WALKING"])
        elif mot == "Bicycle":
            res.update(["ON_BICYCLE", "ELECTRIC_BIKE", "SCOOTER"])
        elif mot == "Bus":
            res.update(["BUS", "ELECTRIC_BUS", "COACH"])
        elif mot == "Tram":
            res.update(["TRAM"])
        elif mot == "Plane":
            res.update(["PLANE"])
        elif mot == "Boat":
            res.update(["BOAT", "BOAT_NO_ENGINE"])
    return list(res)


def merge_consecutive_movements(df, time_min):
    df = df.sort_values(['participant_id', 'mean_of_transport', 'start_time'])
    df['prev_end_time'] = df.groupby(['participant_id', 'mean_of_transport'])['end_time'].shift()
    df['time_diff'] = (df['start_time'] - pd.to_datetime(df['prev_end_time'])).dt.total_seconds() / 60
    df['group'] = (df['time_diff'] > time_min).cumsum()
    df = df.groupby(['participant_id', 'mean_of_transport', 'group']).agg(
        start_time=('start_time', 'first'),
        end_time=('end_time', 'last'),
        start_geohash=('start_geohash', 'first'),
        end_geohash=('end_geohash', 'last'),
        distance=('distance(m)', 'sum'),
        gCO2=('gCO2', 'sum')
    ).reset_index()
    df.rename(columns={'distance': 'distance(m)'}, inplace=True)
    return df


# ------------------- LOAD DATA -------------------
uploaded = st.sidebar.file_uploader("Upload movements CSV", type="csv")

if uploaded:
    df = load_data(uploaded)
else:
    try:
        df = load_data("../data/Test_movements.csv")
    except:
        try:
            df = load_data("./data/Test_movements.csv")
        except:
            st.warning("Please upload a CSV file containing the movements data.")
            st.stop()

# ------------------- FILTERS -------------------
st.sidebar.header("Filters")

line_width = st.sidebar.slider("Line Width (visual only)", 0.1, 5.0, 1.0)
alpha_perc = st.sidebar.slider("Line Alpha", 0.0, 1.0, 0.5)
rand_factor = st.sidebar.slider("Randomness of Line Positions", 0, 100, 10) * 0.0001
min_distance = st.sidebar.slider("Minimum Distance (km)", 0, 150, 0)
max_distance = st.sidebar.slider("Maximum Distance (m)", 0, 10000, 0)
max_time = st.sidebar.slider("Maximum time between movements to merge them (minutes)", 0, 120, 0)

start_date = st.sidebar.date_input("Start Date", min_value=min(df['start_time']), max_value=max(df['start_time']), value=min(df['start_time']))
end_date = st.sidebar.date_input("End Date", min_value=min(df['start_time']), max_value=max(df['start_time']), value=max(df['start_time']))

transport_modes = ["Car", "Train", "Walking", "Bicycle", "Bus", "Tram", "Plane", "Boat"]
selected_modes = translate_mot(st.sidebar.multiselect("Select Transport Modes", transport_modes, default=transport_modes))

participants = df['participant_id'].unique()
selected_participants = st.sidebar.multiselect("Select Participants", participants, default=participants)

# ------------------- FILTER LOGIC -------------------
df = merge_consecutive_movements(df, max_time)
df = df[df['distance(m)'] >= min_distance * 1000]
if max_distance > 0:
    df = df[df['distance(m)'] <= max_distance]
df = df[df['participant_id'].isin(selected_participants)]
df = df[df['start_time'].dt.date >= start_date]
df = df[df['start_time'].dt.date <= end_date]
df = df[df['mean_of_transport'].isin(selected_modes)]

st.sidebar.write(f"Number of rows: {df.shape[0]}")

# ------------------- COORDINATES -------------------
df['start_coords'] = df['start_geohash'].map(geohash_to_coordinate)
df['end_coords'] = df['end_geohash'].map(geohash_to_coordinate)
rand_offsets = (np.random.rand(len(df), 2) - 0.5) * rand_factor

df['start_lat'] = [x[0] + r[0] for x, r in zip(df['start_coords'], rand_offsets)]
df['start_lon'] = [x[1] + r[1] for x, r in zip(df['start_coords'], rand_offsets)]
df['end_lat'] = [x[0] + r[0] for x, r in zip(df['end_coords'], rand_offsets)]
df['end_lon'] = [x[1] + r[1] for x, r in zip(df['end_coords'], rand_offsets)]

df['x0'], df['y0'] = lnglat_to_meters(df['start_lon'], df['start_lat'])
df['x1'], df['y1'] = lnglat_to_meters(df['end_lon'], df['end_lat'])

# ------------------- DATASHADER MAP -------------------
st.markdown("## Transport Movements (Datashader)")

# Make a simple line plot (hv.Curve doesn’t support segments, so we use hv.Segments)
segments = hv.Segments(df, kdims=['x0', 'y0', 'x1', 'y1'])

# Apply datashading for fast rendering
shaded = datashade(
    segments,
    cmap=colorcet.fire,
    aggregator=ds.count(),
    width=1200,
    height=800,
)

# Display in Streamlit
streamlit_bokeh(hv.render(shaded, backend='bokeh'), use_container_width=True)

# ------------------- LEGEND -------------------
st.sidebar.markdown("### Legend")
legend_items = {
    "Walking": "rgba(70,130,180,0.7)",
    "Bicycle / Scooter": "rgba(100,149,237,0.7)",
    "Train": "rgba(0,191,255,0.7)",
    "Tram": "rgba(255,160,122,0.7)",
    "Bus / Coach": "rgba(255,182,193,0.7)",
    "Car / Plane / Boat": "rgba(255,105,180,0.7)",
}
for label, color in legend_items.items():
    st.sidebar.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<div style='background-color: {color}; width: 20px; height: 10px; margin-right: 8px;'></div>{label}</div>",
        unsafe_allow_html=True
    )
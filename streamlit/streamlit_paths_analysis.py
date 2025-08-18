import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import linear
import json
import math
import random
import numpy as np
import pydeck as pdk
import streamlit as st
import matplotlib.pyplot as plt
from geolib import geohash as geolib
from matplotlib import colormaps as cmaps


# Setup
# Define colors for each mode of transport
st.set_page_config(layout="wide")
# Inject custom CSS to make Pydeck fill the full height
st.markdown(
    """
    <style>
        .stDeckGlJsonChart {
            height: calc(100vh - 100px) !important;  /* Adjusts for full viewport height */
            min-height: 600px; /* Ensures a minimum height */
        }
        
        #deckgl-wrapper {
            height: calc(100vh - 100px) !important;  /* Adjusts for full viewport height */
            min-height: 600px; /* Ensures a minimum height
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Swiss boundaries
swiss_boundaries = {
    "min_lat": 45.81922927350267,
    "min_lon": 5.955963134765625,
    "max_lon": 10.492286682128906, 
    "max_lat": 47.80848549672742
}

# Geohashes to coordinates, from file if it exists, otherwise create it
try:
    with open('geohashes_to_coords.json', 'r') as f:
        geo_to_coords = json.load(f)
except:
    geo_to_coords = {}

## Get coordinates from geohashes
def geohash_to_coordinate(geohash):
    try:
        lat, lon = geolib.decode(geohash)
        #lat = float(lat) + 0.00000001#(random.random() - 0.5) * 0.00000001
        #lon = float(lon) + 0.00000001#(random.random() - 0.5) * 0.00000001
        return [float(lat), float(lon)]
    except:
        return [0.0, 0.0]

# Get coordinates from geohashes
@st.cache_data
def geohashes_to_coordinate(geohashes):
    # Get coordinates for each geohash and store in dictionary
    new_data = False
    for geohash in geohashes:
        if geohash not in geo_to_coords:
            new_data = True
            geo_to_coords[geohash] = geohash_to_coordinate(geohash)

    # Save the dictionary to file
    if new_data:
        with open('geohashes_to_coords.json', 'w') as f:
            json.dump(geo_to_coords, f)

    return geo_to_coords
    


# Read the data from the csv
@st.cache_data
def load_data(path):
    # Read the data from the csv
    df = pd.read_csv(path, sep=";")

    # Transform month and time to int
    df['month'] = df['month'].astype(int)
    
    # Get first part of time range
    df['hour'] = df['time_range'].apply(lambda x: x.split('-')[0])
    df['hour'] = df['hour'].astype(int)

    # Group by geohash, month, hour and mode of transport, and count the number of occurrences -- faster for the viz later
    agg = df.groupby(['geohash', 'month', 'hour', 'mode_of_transport']).size().reset_index(name='count')

    return agg

# Filter to reduce the precision of geohashes
def reduce_precision(df, precision):
    df['geohash'] = df['geohash'].apply(lambda x: x[:precision])
    return df

# Translate selected modes of transport to corresponding actual transport modes
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

# Define the colormap
@st.cache_resource
def set_colormap(cm_name='plasma'):
    base = cmaps[cm_name]
    # Define plasma color map
    plasma_colormap = [base(i)[:3] for i in range(256)]  # Get 256 RGB values
    plasma_colormap = [[int(r*255), int(g*255), int(b*255)] for r, g, b in plasma_colormap]  # Convert to 0-255
    return plasma_colormap


# Load data
uploaded = st.sidebar.file_uploader("Upload paths CSV", type="csv")
ok = False
if uploaded:
    df = load_data(uploaded)
    ok = True
else:
    try:
        df = load_data('../data/pathpoints.csv')
        ok = True
    except Exception as e:
        try:
            df = load_data('swice_analysis/data/pathpoints.csv')
            ok = True
        except Exception as e:
            st.error("Error loading paths data." + str(e))
            ok = False
            st.warning("Please upload a CSV file containing the paths data.")

if ok:
    #### FILTERS ####
    # Sidebar filters
    st.sidebar.header("Filters")

    # Select month range
    selected_months = st.sidebar.slider("Select Month Range", min_value=1, max_value=12, value=(1, 12))
    selected_months = list(range(selected_months[0], selected_months[1] + 1))

    # Select time range
    selected_times = st.sidebar.slider("Select Time Range", min_value=0, max_value=23, value=(0, 23))
    selected_times = list(range(selected_times[0], selected_times[1] + 1))

    # Select transport modes
    transport_modes = ["Car", "Train", "Walking", "Bicycle", "Bus", "Tram", "Plane", "Boat"]
    selected_modes = st.sidebar.multiselect("Select Transport Modes", transport_modes, default=transport_modes)
    selected_modes = translate_mot(selected_modes)

    # Only switzerland
    isin_switzerland = st.sidebar.checkbox("Only Switzerland", value=True)

    # Geohash precision
    geohash_precision = st.sidebar.slider("Geohash Precision", 6, 9, 9)

    # Filter data
    df = reduce_precision(df, geohash_precision)

    # Populate the geohash to coordinates dictionary
    geo_to_coords = geohashes_to_coordinate(df['geohash'].unique())

    df = df[df['month'].isin(selected_months)]

    df = df[df['hour'].isin(selected_times)]

    df = df[df['mode_of_transport'].isin(selected_modes)]

    df = df.groupby('geohash', as_index=False)['count'].sum()

    # Transform geohashes to coordinates
    df['coords'] = df['geohash'].map(geo_to_coords)
    df[['latitude', 'longitude']] = pd.DataFrame(df['coords'].tolist(), index=df.index)

    if isin_switzerland:
        df = df[(df['latitude'] >= swiss_boundaries['min_lat']) & (df['latitude'] <= swiss_boundaries['max_lat']) & 
                (df['longitude'] >= swiss_boundaries['min_lon']) & (df['longitude'] <= swiss_boundaries['max_lon'])]

    # Choose colormap based on user selection
    colormap_name = st.sidebar.selectbox("Select Colormap", ["plasma", "coolwarm", "copper", "pink", "bone", "viridis", "magma", "inferno", "cividis"])
    # Set colormap
    plasma_colormap = set_colormap(colormap_name)

    # Change opacity based on user selection
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.7)

    # Show the number of rows in the sidebar
    st.sidebar.write(f"Number of rows: {df.shape[0]}")

    # Prepare data for Pydeck

    # Create a Pydeck layer
    layer = pdk.Layer(
        'HeatmapLayer',
        df,
        get_position=['longitude', 'latitude'],
        get_weight='count',
        color_range=plasma_colormap,  # Apply Plasma colormap
        aggregation = 'SUM',
        opacity=opacity,
        pickable=False
    )

    # Set initial view state based on the initial data
    if 'initial_view_state' not in st.session_state:
        st.session_state.initial_view_state = pdk.ViewState(
            latitude=46.799713,
            longitude=8.235587,
            zoom=8,
            min_zoom=3,
            max_zoom=17,
        )

    view_state = st.session_state.initial_view_state

    st.pydeck_chart(
        pdk.Deck(
            layer,
            initial_view_state=view_state,
            map_style="dark",
        ),
        use_container_width=True
    )
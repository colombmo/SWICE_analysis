import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import linear
from geolib import geohash as geolib
import json
import math
import random
import numpy as np
import pydeck as pdk
import streamlit as st


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


## Get a coordinate from a geohash, adding a small random offset to avoid overlapping
def geohash_to_coordinate(geohash):
    try:
        lat, lon = geolib.decode(geohash)
        #lat = float(lat) + 0.00000001#(random.random() - 0.5) * 0.00000001
        #lon = float(lon) + 0.00000001#(random.random() - 0.5) * 0.00000001
        return [float(lat), float(lon)]
    except:
        return [float(0.0), float(0.0)]

# Read the data from the csv
@st.cache_data
def load_data(path):
    # Read the data from the csv
    df = gpd.read_file(path)

    # Remove timezone
    df['start_time'] = df['start_time'].str[:-6]
    df['end_time'] = df['end_time'].str[:-6]

    # Transform the start_ and end_date to datetimes
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    ## Convert distance to int
    df['distance(m)'] = df['distance(m)'].astype(int)

    ## Convert gCO2 to int
    df['gCO2'] = df['gCO2'].astype(int)

    # IF two movements from the same user have the same start geohash and start time, we should merge them into a single movement (keep the one with the highest distance)
    df['start_time'] = pd.to_datetime(df['start_time'])

    # Group by participant_id, start_geohash, start_time and maximum distance, keep all the other data from the row with the maxium distance
    df = df.groupby(['participant_id', 'start_geohash', 'start_time'])['distance(m)'].max().reset_index().merge(df, on=['participant_id', 'start_geohash', 'start_time', 'distance(m)'])

    return df

# Filter to reduce the precision of long movements
def reduce_precision(df, min_dist, precision):
    # Reduce precision of geohash by 1 character for the rows where we covered a long distance (more than 10 km)
    df['start_geohash'] = df.apply(lambda x: x['start_geohash'][:precision] if x['distance(m)'] > min_dist else x['start_geohash'], axis=1)
    df['end_geohash'] = df.apply(lambda x: x['end_geohash'][:precision] if x['distance(m)'] > min_dist else x['end_geohash'], axis=1)
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


# Merge consecutive movements with the same mode of transport
def merge_consecutive_movements(df, time_min):
    # Group movements by participant and mode of transport, and sort by start time
    df = df.sort_values(['participant_id', 'mean_of_transport', 'start_time'])

    # Calculate time difference between end time of one movement and start time of the next
    df['prev_end_time'] = df.groupby(['participant_id', 'mean_of_transport'])['end_time'].shift()
    df['time_diff'] = (df['start_time'] - pd.to_datetime(df['prev_end_time'])).dt.total_seconds() / 60

    # Create a new column to store the group number
    df['group'] = (df['time_diff'] > time_min).cumsum()

    # Group by participant, mode of transport and group number, and calculate the start and end time of the group
    df = df.groupby(['participant_id', 'mean_of_transport', 'group']).agg(
        start_time=('start_time', 'first'),
        end_time=('end_time', 'last'),
        start_geohash=('start_geohash', 'first'),
        end_geohash=('end_geohash', 'last'),
        distance=('distance(m)', 'sum'),
        gCO2=('gCO2', 'sum')
    ).reset_index()

    # Rename columns to ensure consistency - For some reason groupby changes distance(m) to distance...!
    df.rename(columns={'distance': 'distance(m)'}, inplace=True)

    return df


# Load data
df = load_data('../data/Test_movements.csv')

#### FILTERS ####
# Sidebar filters
st.sidebar.header("Filters")

# Width of lines
line_width = st.sidebar.slider("Line Width", 0.1, 5.0, 1.0)

# Transparency of lines
alpha_perc = st.sidebar.slider("Line Alpha", 0.0, 1.0, 0.5)

# Randomness of line positions
rand_factor = st.sidebar.slider("Randomness of Line Positions", 0, 100, 10)*0.0001

# Minimum length of trips
min_distance = st.sidebar.slider("Minimum Distance (km)", 0, 150, 0)

# maxium length of trips
max_distance = st.sidebar.slider("Maximum Distance (m)", 0, 10000, 0)

# Geohash precision for long trips
geohash_precision = st.sidebar.slider("Geohash Precision for long trips", 4, 6, 6)

# Maximum time between movements to merge them, whgen they have the same mode of transport
max_time = st.sidebar.slider("Maximum time between movements to merge them (minutes)", 0, 120, 0)

# Start date range
start_date = st.sidebar.date_input("Start Date", min_value=min(df['start_time']), max_value=max(df['start_time']), value=min(df['start_time']))

transport_modes = ["Car", "Train", "Walking", "Bicycle", "Bus", "Tram", "Plane", "Boat"]
selected_modes = st.sidebar.multiselect("Select Transport Modes", transport_modes, default=transport_modes)
selected_modes = translate_mot(selected_modes)

# Participants
participants = df['participant_id'].unique()
selected_participants = st.sidebar.multiselect("Select Participants", participants, default=participants)

##################
alpha = int(255 * alpha_perc)
colors = {
    'WALKING': [70, 130, 180, alpha],      # Steel Blue (distinct blue for walking)
    'ON_BICYCLE': [100, 149, 237, alpha],  # Cornflower Blue (different blue for bike)
    'ELECTRIC_BIKE': [100, 149, 237, alpha],
    'SCOOTER': [100, 149, 237, alpha],
    'TRAIN': [0, 191, 255, alpha],         # Deep Sky Blue (different blue for train)
    'BUS': [255, 182, 193, alpha],         # Light Pink (distinct from car/tram/plane)
    'ELECTRIC_BUS': [255, 182, 193, alpha],
    'COACH': [255, 182, 193, alpha],
    'CAR': [255, 105, 180, alpha],        # Hot Pink (distinct for car)
    'ELECTRIC_CAR': [255, 105, 180, alpha],
    'HYBRID_CAR': [255, 105, 180, alpha],
    'TRAM': [255, 160, 122, alpha],       # Light Salmon Pink (distinct from bus and car)
    'PLANE': [255, 105, 180, alpha],      # Hot Pink (same as car, but could be changed if needed)
    'BOAT': [255, 105, 180, alpha],       # Hot Pink
    'BOAT_NO_ENGINE': [255, 105, 180, alpha],
    'DETECTION_ERROR': [0, 0, 0, alpha],  # Black
}

# Filter data
df = merge_consecutive_movements(df, max_time)

df = df[df['distance(m)'] >= min_distance * 1000]

if max_distance > 0:
    df = df[df['distance(m)'] <= max_distance]

df = df[df['participant_id'].isin(selected_participants)]

df = df[df['start_time'].dt.date >= start_date]

df = df[df['mean_of_transport'].isin(selected_modes)]

df = reduce_precision(df, 10000, geohash_precision)

# Show the number of rows in the sidebar
st.sidebar.write(f"Number of rows: {df.shape[0]}")

# Prepare data for Pydeck
# Transform geohashes to coordinates
df['start_coords'] = df['start_geohash'].apply(geohash_to_coordinate)
df['end_coords'] = df['end_geohash'].apply(geohash_to_coordinate)

# Add slight offsets to avoid overlaps
df['start_coords'] = df['start_coords'].apply(lambda x: [x[1] + (random.random() - 0.5) * rand_factor, x[0] + (random.random() - 0.5) * rand_factor])
df['end_coords'] = df['end_coords'].apply(lambda x: [x[1] + (random.random() - 0.5) * rand_factor, x[0] + (random.random() - 0.5) * rand_factor])


# Add random height to the start and end coordinates
#df['start_coords'] = df['start_coords'].apply(lambda x: [x[0], x[1], random.randint(10000, 20000)])
#df['end_coords'] = df['end_coords'].apply(lambda x: [x[0], x[1], random.randint(10000, 20000)])

# Create data list for Pydeck
path_data = df[['start_coords', 'end_coords', 'mean_of_transport', 'participant_id']].copy()
path_data['color'] = path_data['mean_of_transport'].apply(lambda x: colors.get(x, [0, 0, 0])) # Default black for unknown transport
path_data['color_start'] = path_data['color'].apply(lambda x: [xi * 0.7 for xi in x])  # Slightly darker starting 


# Pydeck Layer
layer = pdk.Layer(
    "ArcLayer",  # Change to "LineLayer" if needed
    data=path_data,
    get_source_position="start_coords",
    get_target_position="end_coords",
    get_source_color="color_start",
    get_target_color="color",
    get_width=line_width,
    pickable=True,
    auto_highlight=True,
)

# Define View
view_state = pdk.ViewState(
    latitude=df['start_coords'].iloc[0][1],
    longitude=df['start_coords'].iloc[0][0],
    zoom=8,
    pitch=30,
)

# Render map
st.pydeck_chart(
    pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state, 
        tooltip={"text": "{participant_id} - {mean_of_transport}"}
        ),
        use_container_width=True,
    )


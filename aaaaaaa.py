import streamlit as st

transport_modes = ["Car", "Train", "Walking", "Bicycle", "Bus", "Tram", "Plane", "Boat"]
selected_mode = st.sidebar.selectbox("Select Transport Mode", transport_modes, index=0) # Or checkbox

st.write("You selected the following transport mode:")
st.write(f"- {selected_mode}")

if selected_mode == "Car":
    # Read html content from the car visualization file
    html = io.read("car_visualization.html")
elif selected_mode == "Train":
    # Read html content from the train visualization file
    html = io.read("train_visualization.html")
else:
    # Show all modes of transport
    html = io.read("all_transport_visualization.html")


st.html(html)
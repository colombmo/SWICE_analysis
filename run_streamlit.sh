source osmnx/bin/activate
cd streamlit

# Run streamlit app. We can define if to show movements or paths with a variable
if [ "$1" = "movements" ]; then
    streamlit run streamlit_movements_analysis.py
elif [ "$1" = "paths" ]; then
    streamlit run streamlit_paths_analysis.py
elif [ "$1" = "datashader" ]; then
    streamlit run streamlit_datashader_moves.py
else
    echo "Invalid argument. Please use 'movements', 'paths' or 'datashader'"
fi
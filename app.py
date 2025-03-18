import streamlit as st
import tc
import tr

# Set page config
st.set_page_config(
    page_title="Business Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, sleek design
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stSelectbox {
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 0.3rem;
        border: none;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .plot-container {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Business Forecasting Dashboard")

# Business selector
business = st.selectbox(
    "Select Business",
    ["Travellers Cavern", "Travelicious Restaurant"],
    index=0
)

# Forecast button
if st.button("Generate Forecast"):
    if business == "Travellers Cavern":
        tc.show_dashboard()
    else:
        tr.show_dashboard()

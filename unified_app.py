import streamlit as st

# Load CSS file we created
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Light/Dark Mode Selection
theme_selection = st.sidebar.selectbox("Select Theme:", ["Dark Mode", "Light Mode"])

# Determine the theme class based on selection
theme_class = "dark-theme" if theme_selection == "Dark Mode" else "light-theme"


st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 3em;'>Welcome to the Stock Price Prediction App</h1>", unsafe_allow_html=True)


tabs = st.tabs(["Comprehensive Analysis", "Stock Prediction", "Forecast"])
with tabs[0]:
    exec(open("app.py").read())
with tabs[1]:
    exec(open("combined_app.py").read())
with tabs[2]:
    exec(open("main.py").read())

st.markdown("</div>", unsafe_allow_html=True)

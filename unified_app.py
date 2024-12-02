import streamlit as st

# Light/Dark Mode Toggle
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

if st.sidebar.button("Toggle Theme"):
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"

theme_class = "dark-theme" if st.session_state["theme"] == "dark" else "light-theme"
st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True)

# Tabs for Navigation
tabs = st.tabs(["Comprehensive Analysis", "Stock Prediction", "Forecast"])
with tabs[0]:
    exec(open("app.py").read())
with tabs[1]:
    exec(open("combined_app.py").read())
with tabs[2]:
    exec(open("main.py").read())

st.markdown("</div>", unsafe_allow_html=True)

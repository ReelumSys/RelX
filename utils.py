import streamlit as st
from streamlit_extras.app_logo import add_logo

def draw_something_on_top_of_page_navigation():
    st.sidebar.markdown(
        "My Logo (sidebar) should be on top of the Navigation within the sidebar"
    )

def logo():
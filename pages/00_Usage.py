# Contents of ~/my_app/pages/page_0.py
import streamlit as st
import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
from PIL import Image

im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("../RelX/images/favicon.png");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


#st.markdown("### Acknowledgement")
st.write("###### With this app it is possible to do some work on XRD Charts with only a very few amount of data. Since for Rietveld Refinement only atomic parameters and HKL-values are needed. Same for the Bravais crystal structure.")
st.text("")
st.write("###### The Charts section is also for comparing data. The Main gets subtracted by the Comparing.")

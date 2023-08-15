# Contents of ~/my_app/pages/page_2.py
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
    background-image: url("../RelX/images/favicon2.png");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


#st.markdown("### Acknowledgement")
st.write("###### Thanks to PowerXRDs work by Andrew Garcia Ph.D (https://github.com/andrewrgarcia/powerxrd) a Williamson-Hall-Plot and other functions were written. The Williamson-Hall-Plot is based on the same values as the Scherrer calculation. Furthermore a Halder and Wagner-Plot is made possible from the Literature. The plots should serve as a direct comparison between the three methods.")
st.text("")
st.write("First you can upload two XRD charts. They need not to have the same 2Theta values as you can set a beginning value. However the patterns have to be recorded at same diffraction settings. The main runs Crystal Size and Rietveld Refinement. After uploading, first you see a heatmap plus indentified and binned values. This should help with the data conformity. Beside the heatmaps you can find an analysis of data in the donut chart. As you loaded two charts, the backsubbed area from PowerXRD is read. In the Charts section there are three plots where the last plot is a subtraction between the Main XRD and the Comparison XRD data. After the three, the same procedure is done with in logarithmic scale for the intensity. This should provide an easy access.")
st.text("")
st.write("The Rietveld equation and the parameters were modified. However lmfit does a bad job here as its purpose is to fit through multiple point close together at the x-Axis. Ongoing work is done here. Need a conversion to a line graph and still run all parameters. Or you are welcome to try with a very long diffraction time.")
st.text("")
st.write("##### Literature")
st.write("W. H. Hall. (1949) Proc. Phys. Soc. A (London, U. K.) 62 741–743.")
st.write("Izumi, F., Ikeda, T. (2014) - Implementation of the Williamson–Hall and Halder and Wagner Methods into RIETAN-FP")

#st.sidebar.markdown("##### vladimirvopravil@hotmail.com")


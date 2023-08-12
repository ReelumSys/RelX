import streamlit as st
import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
import Main
from WH import m
from WH import d
from WH import*
from PIL import Image


im = Image.open("../Relx/favicon2.png")
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('')




st.markdown('### Main')
x, y = xrd.Data('ksev1.xy').importfile()
model = xrd.Rietveld(x, y)
model.refine()


uploaded_file5 = st.image("RietveldRef.png")
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
from WH import Ee2
from WH import r
import urllib
import urllib3



im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('')

uploaded_file = ('FHWMFirstSecond.csv')

data = pd.read_csv(uploaded_file, sep=" ", names=['Int','Scherrer'])
Cryst5 = data['Scherrer'].mean()

st.write("Crystal Size and Strain are all calculated with a gaussian refinement.")

Cryst = pd.DataFrame({
                      
                      'Scherrer Size [nm]': [Cryst5],
                      'W-H Size [nm]': [d],
                      'W-H Strain [%%]': [m],
                      
                      'H-W Size [nm]': [Ee2],
                      'H-W Strain [%%]': [r],
                    })

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)
#Cryst['Type'] = ['Crystal Strain', 'Crystal Size after WH' ]
st.markdown('##### Main')
st.table(data=Cryst)

st.markdown('###### W-H Plot')
uploaded_file = st.image("WH-PLOT.png")
st.markdown('###### H-W Plot')
uploaded_file = st.image("HW-PLOT.png")

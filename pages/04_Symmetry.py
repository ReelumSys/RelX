from PyCrystallography import unit_cell
from PyCrystallography import lattice
from PyCrystallography import unit_cell


import matplotlib .pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
import Main
import Main as StValv
from PIL import Image

global weather1
global weather2
global weather3
#im = Image.open("../Relx/favicon2.png")
st.set_page_config(
    page_title="RelX v0.9",

    #page_icon=im,
    layout="wide",
)


#global StValv
#StValv =20
#print(StValv)
#StartingValue = StValv

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('')


st.markdown('##### Basic Cells')
uploaded_file2 = st.image("Bravais.png")
st.markdown('##### Bravais & Reciprocal')
uploaded_file3 = st.image("Bravais2.png")
st.markdown('##### 2D Lattice')
uploaded_file6 = st.image("Bravais5.png")
st.markdown('##### 3D Lattice')

uploaded_file4 = st.image("Bravais3.png")
st.markdown('##### Cuboid')
uploaded_file5 = st.image("Bravais4.png")





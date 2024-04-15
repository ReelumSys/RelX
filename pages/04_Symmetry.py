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
from pathlib import Path

global weather1
global weather2
global weather3

im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)


#global StValv
#StValv =20
#print(StValv)
#StartingValue = StValv

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('')


my_file = Path("HKL.csv")
try:
    my_abs_path = my_file.resolve(strict=True)
except FileNotFoundError:
    uploaded_file3 = st.file_uploader("Upload a .txt of the HKLs for Rietveld Refinement and Bravais calculations", type=["txt"])
    st.text("")
    st.write("The HKL should be formatted with a space as delimiter.")
    st.text("")
    image = Image.open('./images/HKLinfo.png')
    new_img = image.resize((125, 250))
    st.image(new_img)
    st.text("")



    name3 = uploaded_file3
    if not name3:
      st.warning('Please input a .txt file.')
      st.stop()
    st.success('Done.')

    df = pd.read_fwf(name3)
    #df.to_csv('HKL.csv', index=None)
    np.savetxt('HKL.csv', df, fmt='%i', delimiter=',')
    np.savetxt('HKL.txt', df, fmt='%i', delimiter=' ')

    
    
    #st.markdown('##### Basic Cells')
    #uploaded_file2 = st.image("Bravais.png")
    #st.markdown('##### Bravais & Reciprocal')
    #uploaded_file3 = st.image("Bravais2.png")
    #st.markdown('##### 2D Lattice')
    #uploaded_file6 = st.image("Bravais5.png")
    #st.markdown('##### 3D Lattice')

    #uploaded_file4 = st.image("Bravais3.png")
    #st.markdown('##### Cuboid')
    #uploaded_file5 = st.image("Bravais4.png")
else:
    # exists



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





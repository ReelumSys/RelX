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
from powerxrd import AP1


im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('')






  
#number = st.number_input('Enter HKL')
#st.write('The current number is ', number)
#number = st.number_input("Insert a number", value=None, placeholder="A number between 0 and 1", min_value=0, max_value=1)
#st.write('The current number is ', number)



st.markdown('### Main')
x, y = xrd.Data('ksev1.xy').importfile()
model = xrd.Rietveld(x, y)
model.refine()


uploaded_file5 = st.image("RietveldRef.png")

#df = pd.read_fwf('HKL.csv')

#dfHKL = pd.read_fwf('HKL.csv', header=False)
global dfHKL
dfHKL = pd.read_csv('HKL.csv', names=['H','K','L'], sep=',', index_col=None)

#dfHKL['index1'] = dfHKL.index
 

st.dataframe(dfHKL)
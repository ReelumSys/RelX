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
from powerxrd import main



global AtomAinX
global AtomAinY
global AtomAinZ


im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('')

st.markdown("## ")
st.sidebar.markdown("# ")

#st.sidebar.subheader('Heat map parameter')
#time_hist_color = st.sidebar.selectbox('Color by', '')
#st.sidebar.subheader('Atom A')
#AtomAinX = st.sidebar.number_input(label='For X in Atom A',format="%.2f")
#AtomAinY = st.sidebar.number_input(label='For Y in Atom A',format="%.2f")
#AtomAinZ = st.sidebar.number_input(label='For Z in Atom A',format="%.2f")


global AP1
global AP2
global AP3



  
#AP1 = st.number_input('Enter x for Atom A',format="%f")
#st.write('The current number is ', number)
#AP1 = st.number_input("Insert a number", placeholder="A number between 0 and 1", step=1, format="%f")
#st.write('The current number is ', AP1)

#AP2 = st.number_input('Enter y for Atom A',format="%f")
#st.write('The current number is ', number)
#AP1 = st.number_input("Insert a number", placeholder="A number between 0 and 1", step=1, format="%f")
#st.write('The current number is ', AP2)

#AP3 = st.number_input('Enter y for Atom A',format="%f")
#st.write('The current number is ', number)
#AP1 = st.number_input("Insert a number", placeholder="A number between 0 and 1", step=1, format="%f")
#st.write('The current number is ', AP3)


uploaded_file = st.file_uploader("Upload atomic coordinates as .txt like in the example with with space as delimiter.", type=["txt"])

st.text("")
image = Image.open('./images/AtomCoordexample.png')
new_img = image.resize((250, 150))
st.image(new_img)
st.text("")

name = uploaded_file
if not name:
  st.warning('Please input a .txt file.')
  st.stop()
st.success('Done.')

df = pd.read_fwf(name)
df.to_csv('AtomicCoordinates.csv', index=False)
np.savetxt('AtomicCoordinates2.csv', df, fmt='%f', delimiter=',')

uploaded_file2 = st.file_uploader("Upload Atomic Displacement values if needed as .txt like in the example.", type=["txt"])
#Atomic_Displacement1 = st.number_input(label="Atomic Displacement",format="%.2f") 
st.text("")
image = Image.open('./images/AtomDis.png')
new_img = image.resize((180, 120))
st.image(new_img)
st.text("")

name2 = uploaded_file2
if not name2:
  st.warning('Please input a .txt file.')
  st.stop()
st.success('Done.')

df = pd.read_fwf(name2)
df.to_csv('AtomicDisplacement.csv', index=False)
np.savetxt('AtomicDisplacement2.csv', df, fmt='%f', delimiter=',')

name3 = st.number_input(label="Scale Factor",step=0.0000000001,format="%f") 
if not name3:
  st.warning('Please input a number.')
  st.stop()
st.success('Done.')

df = name3
df.to_csv('ScaleFactor.csv', index=False)
np.savetxt('ScaleFactor2.csv', df, fmt='%f', delimiter=',')

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
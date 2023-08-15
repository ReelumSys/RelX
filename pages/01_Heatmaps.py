import streamlit as st
import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
import Main

# Contents of ~/my_app/main_page.py


from PIL import Image



im = 'favicon.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("## ")
st.sidebar.markdown("# ")

#st.sidebar.subheader('Heat map parameter')
#time_hist_color = st.sidebar.selectbox('Color by', '')
st.sidebar.subheader('Donut chart parameter')
donut_theta = st.sidebar.selectbox('Select data', ('Theta', 'Area2'))

dfheat = pd.read_csv('ksev1.csv', names=['2Theta','Int'])
stocks = pd.read_csv('Area.csv')

c1, c2 = st.columns((2,1))
with c1:
    st.markdown('#### Main vs. Comp XRD')
    plost.scatter_hist(
        data=dfheat,
        x='2Theta',
        y='Int',
        size='Int',
        color='Int',
        opacity=0.5,
        aggregate='count',
        width=200,
        height=200,
        legend='bottom',
        use_container_width=True
    )
    
    st.markdown('#### Data Main XRD')
    st.markdown('###### Check if all fields match')
    plost.xy_hist(
        data=dfheat,
        x='2Theta',
        y='Int',
        #x_bin=100,
        #y_bin='Int2',
        use_container_width=True,
    )
    

with c2:
    st.markdown('### XRDs vs. in %')
    plost.donut_chart(
        data=stocks,
        #theta=donut_theta,
        theta='Area',
        color='Sample',
        
        legend='bottom', 
        use_container_width=True)

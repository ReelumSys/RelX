# Contents of ~/my_app/main_page.py

import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
from scipy.integrate import simpson
from numpy import trapz
import cv2

import powerxrd as xrd
import numpy as np
import pandas as pd
import io
from matplotlib import* #pylab as plt
import matplotlib.pyplot as plt
#import subprocess
import sys
sys.path.append('../powerxrd/powerxrd/powerxrd')
import csv
from itertools import repeat
import os
import time

from numpy import*
from scipy import*

from scipy.optimize import fmin
from scipy.optimize import curve_fit

from powerxrd.main import scherrer as beta
from powerxrd.main import Chart as SchPeak
from powerxrd.main import Chart as SchPeak
from powerxrd.main import*
import contextlib
from WH import m
from WH import d
from WH import*
import acknowledgement
import main


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('')

#plot_height = st.sidebar.slider('Specify plot height', 200, 1000, 250)
#st.button("Next Page")
st.sidebar.markdown('''
---

''')



#uploaded_file2 = st.image("RietveldRef.png")
uploaded_file = st.file_uploader("Choose Main File")
dataframe = pd.read_csv(uploaded_file)

#uploaded_file2 = st.image("RietveldRef.png")
uploaded_file = st.file_uploader("Choose Comparing File")
dataframe = pd.read_csv(uploaded_file)
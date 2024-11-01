﻿# Contents of ~/my_app/main_page.py

import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
from scipy.integrate import simpson
from numpy import trapz
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from PyCrystallography import unit_cell
from PyCrystallography import lattice



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
#sys.path.append("path to the xrayutilities package")
import xrayutilities as xu

import utils as ut





from numpy import*
from scipy import*
import urllib

from scipy.optimize import fmin
from scipy.optimize import curve_fit


import contextlib

#from WH import*

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyCrystallography.geometry import *
from PyCrystallography.stereographic_projection import*

import numpy as np
from lattpy import Lattice
from lattpy import simple_square
from utils import logo
#from st_pages import show_pages, hide_pages, Page



    # Clear values from *all* all in-memory and on-disk data caches:
    # i.e. clear values from both square and cube
#st.cache_data.clear()

#from ... import Charts as weather1
#from ... import Charts as weather2
#from ... import Charts as weather3

#StValv = 20
#global StValv
StartingValue = 10




im = Image.open("favicon2.png")
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)
#st.sidebar.image("./images/favicon.png", width=150)


image = Image.open('./images/favicon.png')
new_img = image.resize((180, 100))
#st.image(new_img)
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(new_img)


#st.header('Home')

# Sidebar navigation
#st.sidebar.page_link('Main.py', label='Home')
st.sidebar.page_link('pages/00_Usage.py', label='Usage')
st.sidebar.page_link('pages/01_Comparison.py', label='Comparison')
st.sidebar.page_link('pages/02_Patterns.py', label='Patterns')
st.sidebar.page_link('pages/04_Symmetry.py', label='Symmetry')
st.sidebar.page_link('pages/05_Crystal_Size_&_Strain.py', label='Crystal Size & Strain')


st.sidebar.page_link('pages/08_Acknowledgements.py', label='Acknowledgements')


#image = Image.open('./images/favicon.png')
#new_img = image.resize((200, 100))
#st.image(new_img)
#st.sidebar.markdown("# Main page")
st.markdown("#### Upload XRD patterns and calculate")
st.text("")
st.write("Like in this example, the sample should be delimited with a space. Decimals do not matter.")
st.text("")



image = Image.open('./images/Unbenannt.png')
background = Image.open("./images/Unbenannt.png")
col1, col2, col3 = st.columns([2, 5, 0.2])
col2.image(background, use_column_width=False)



#left_co, cent_co,last_co = st.columns(3)
#with cent_co:
#    st.image(image)


#new_img = image.resize((200, 220))
#st.image(image)
st.text("")
st.markdown('###### Upload two .txt patterns separately and let them be calculated.')

# Allow only .csv and .xlsx files to be uploaded
uploaded_file = st.file_uploader("Upload Main XRD Pattern", type=["txt"])

name = uploaded_file
if not name:
  st.warning('Please input a .txt file.')
  st.stop()
st.success('Done.')



uploaded_file2 = st.file_uploader("Upload Compairing XRD Pattern", type=["txt"])

name2 = uploaded_file2
if not name2:
  st.warning('Please input a .txt file.')

  st.stop()
st.success('Done.')

df = pd.read_fwf(name)
df.to_csv('crash1.csv', index=False)

np.savetxt('crash1.xy', df, fmt='%f', delimiter='\t')
np.savetxt('crash1.csv', df, fmt='%f', delimiter=',')

df = pd.read_fwf(name2)
df.to_csv('crash2.csv', index=False)
np.savetxt('crash2.csv', df, fmt='%f', delimiter=',')

#uploaded_file3 = st.file_uploader("Upload a .txt of the HKLs for Rietveld Refinement and Bravais calculations", type=["txt"])
#st.text("")
#st.write("The HKL should be formatted with a space as delimiter.")
#st.text("")
#image = Image.open('./images/HKLinfo.png')
#new_img = image.resize((125, 250))
#st.image(new_img)
#st.text("")



#name3 = uploaded_file3
#if not name3:
#  st.warning('Please input a .txt file.')
#  st.stop()
#st.success('Done.')

#df = pd.read_fwf(name3)
#df.to_csv('HKL.csv', index=None)
#np.savetxt('HKL.csv', df, fmt='%i', delimiter=',')
#np.savetxt('HKL.txt', df, fmt='%i', delimiter=' ')


#c = np.array(name3)
#na = c.reshape(1)

#np.savetxt('HKL.csv', na, fmt='%s', delimiter=',')
#print(name3)
#ut.draw_something_on_top_of_page_navigation()

os.system("WH.py")



def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

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

#st.set_page_config(layout='wide', initial_sidebar_state='expanded')


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



    st.sidebar.title("Explore")


st.sidebar.header('')





image = Image.open('./images/favicon.png')
new_img = image.resize((180, 100))
#st.image(new_img)
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(new_img)



st.sidebar.subheader('Line chart parameters')
plot_height = st.sidebar.slider('Specify plot height', 200, 1000, 250)

st.sidebar.markdown('''
---

''')


# Row B
global weather1
global weather2
global weather3

weather1 = pd.read_csv('crash1.csv', names=['2Theta','Int'])
weather2 = pd.read_csv('crash2.csv', names=['2Theta','Int2'])
weather3 = pd.read_csv('testTheta2.txt', names=['2Theta','Diff'])
weather5 = pd.read_csv('WH-realx.txt', names=['sinTheta'])
weather6 = pd.read_csv('WH-realy.txt', names=['BetaCosTheta'])

zett = np.polyfit(weather5['sinTheta'], weather6['BetaCosTheta'], 1)
#pee = np.poly1d(zett)

weather_WH = weather5.join(weather6)
#print(weather_WH)
weather_avg = sum(weather_WH['BetaCosTheta']) / len(weather_WH['BetaCosTheta'])
#print(weather_avg)


integral = np.array([])
for col in weather1.columns[1:]:
    temp = weather1.iloc[:, 1:].apply(lambda x: integrate.trapz(x,weather1['Int']))
    integral = np.append(integral,temp)
#print(integral)
#np.savetxt('Integration1.txt', integral, fmt='%f')

integral2 = np.array([])
for col in weather2.columns[1:]:
    temp = weather2.iloc[:, 1:].apply(lambda x: integrate.trapz(x,weather2['Int2']))
    integral2 = np.append(integral2,temp)


df_app = 'crash1', 'crash2'
#df_app2 = pd.DataFrame({'Theta': df_app})
#print(df_app)

df_merged = [integral, integral2]
#print(df_merged)
np.savetxt('area.txt', df_merged, fmt='%f', delimiter=',')
area = pd.read_csv('area.txt', names=['Area','Sample'])

#np.savetxt('area2.txt', df_merged, fmt='%f', delimiter=',')
#area = pd.read_csv('area2.txt', names=['Area2','Sample'])


area['Sample'] = df_app

columns_titles = ["Sample","Area"]
area=area.reindex(columns=columns_titles)
#print(area)
area.to_csv("Area.csv", index=False)

stocks = pd.read_csv('Area.csv')
stocks.to_csv("Area.csv", index=False)
#print(stocks)


#Inte = trapz(df.iloc[:, 'Theta'], df.iloc[:, 'Int'])
#print(Inte)
#np.savetxt('area.txt', Inte, fmt='%f', delimiter=',')
dfheat = pd.read_csv('crash1.csv', names=['2Theta','Int'])
df2heat = pd.read_csv('crash2.csv', names=['2Theta','Int2'])
dfheat['Int2'] = df2heat['Int2']

df = pd.read_fwf('FWHMFirst.txt', header=None)

np.savetxt('FHWMFirstSecond.csv', df, fmt='%s', delimiter=' ')
#df.to_csv('FWHMFirstSecond.txt')
data = pd.read_csv('FHWMFirstSecond.csv', sep=" ", names=['Int','Scherrer'])




fig = plt.figure(0,figsize=[8,8])
ax = fig.add_subplot(111,projection='3d')

prim = unit_cell.BCC(ax)

fig = plt.figure(1,figsize=[8,8])
ax = fig.add_subplot(111,projection='3d')

lattice.make_lattice_3d(ax,prim)
#plt.show()
plt.savefig("Bravais.png")
plt.close()

#set lattice depths
d = 3

#set primitive vectors
A = [1,0,0]
B = [0,1,0]
C = [0,0,10]

vectors = [A,B,C]

fig = plt.figure('Latiices',figsize=[8,4])
ax1 = fig.add_subplot(121,projection='3d')
ax1.set_title('Bravais')
prim = unit_cell.custom_unit_cell(ax1,vectors)
lattice.make_lattice_3d(ax1,prim,depth=d)

vectors_r = lattice.make_vectors_reciprocal(vectors)

ax2 = fig.add_subplot(122,projection='3d')
ax2.set_title('Reciprocal')
prim = unit_cell.custom_unit_cell(ax2,vectors_r)
lattice.make_lattice_3d(ax2,prim,depth=d,tag='_r')

plt.savefig("Bravais2.png")
plt.close()

fig = plt.figure(0,figsize=[8,8])
ax = fig.add_subplot(111,projection='3d')

prim = unit_cell.Diamond(ax)

fig = plt.figure(1,figsize=[8,8])
ax = fig.add_subplot(111,projection='3d')

lattice.make_lattice_3d(ax,prim)
plt.savefig("Bravais3.png")
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
######################################
# which ever 3d model you want to load
cuboid(ax,5,5,5)
######################################
plt.savefig("Bravais4.png")
plt.close()

latt = Lattice(np.eye(2))                 # Construct a Bravais lattice with square unit-vectors
latt.add_atom(pos=[0.0, 0.0])             # Add an Atom to the unit cell of the lattice
latt.add_connections(1)                   # Set the maximum number of distances between all atoms
 
latt = Lattice(np.eye(2))                 # Construct a Bravais lattice with square unit-vectors
latt.add_atom(pos=[0.48768,0.33330], atom="A") 
latt.add_atom(pos=[0.43868,0.14752], atom="B")  # Add an Atom to the unit cell of the lattice
#latt.add_atom(pos=[0.5, 0.5], atom="B")   # Add an Atom to the unit cell of the lattice
latt.add_connection("A", "A", 1)          # Set the max number of distances between A and A
latt.add_connection("A", "B", 1)          # Set the max number of distances between A and B
latt.add_connection("B", "B", 1)          # Set the max number of distances between B and B
latt.analyze()


latt = simple_square(a=1.0, neighbors=1)  # Initializes a square lattice with one atom in the unit-cel
latt.build(shape=(5, 3))
latt.set_periodic(axis=0)
latt.plot()
plt.savefig("Bravais5.png")
plt.close()

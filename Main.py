# Contents of ~/my_app/main_page.py

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
import WH

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
import urllib

from scipy.optimize import fmin
from scipy.optimize import curve_fit

from powerxrd.main import scherrer as beta
from powerxrd.main import Chart as SchPeak

from powerxrd.main import*
import contextlib
from WH import m
from WH import d
#from WH import*

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyCrystallography.geometry import *
from PyCrystallography.stereographic_projection import*

import numpy as np
from lattpy import Lattice
from lattpy import simple_square

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

image = Image.open('./images/favicon.png')
new_img = image.resize((180, 100))
st.image(new_img)

#image = Image.open('./images/favicon.png')
#new_img = image.resize((200, 100))
#st.image(new_img)
#st.sidebar.markdown("# Main page")
st.markdown("#### Upload XRD files and calculate")
st.text("")
st.write("Like this example, the sample should be delimited with a space. The digits do not matter.")
st.text("")
image = Image.open('./images/Unbenannt.png')
new_img = image.resize((220, 220))
st.image(new_img)
st.text("")
st.markdown('###### First upload two .txt files separetely and let them be calculated.')

# Allow only .csv and .xlsx files to be uploaded
uploaded_file = st.file_uploader("Upload Main XRD", type=["txt"])

name = uploaded_file
if not name:
  st.warning('Please input a .txt file.')
  st.stop()
st.success('Done.')



uploaded_file2 = st.file_uploader("Upload Compairing XRD", type=["txt"])

name2 = uploaded_file2
if not name2:
  st.warning('Please input a .csv .txt file.')

  st.stop()
st.success('Done.')

df = pd.read_fwf(name)
df.to_csv('ksev1.csv', index=False)

df = pd.read_fwf(name2)
df.to_csv('ksev1rand.csv', index=False)



os.system("WH.py 1")



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
    
st.sidebar.header('')

image = Image.open('./images/favicon.png')
new_img = image.resize((200, 100))
st.image(new_img)

st.sidebar.subheader('Heat map parameter')
time_hist_color = st.sidebar.selectbox('Color by', '') 

st.sidebar.subheader('Donut chart parameter')
donut_theta = st.sidebar.selectbox('Select data', ('', ''))
#donut_theta = st.sidebar.selectbox('Select data', ('Area'))

st.sidebar.subheader('Line chart parameters')
plot_height = st.sidebar.slider('Specify plot height', 200, 1000, 250)

st.sidebar.markdown('''
---

''')


# Row B
global weather1
global weather2
global weather3

weather1 = pd.read_csv('ksev1.csv', names=['2Theta','Int'])
weather2 = pd.read_csv('ksev1rand.csv', names=['2Theta','Int2'])
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


df_app = 'ksev1', 'ksev1rand'
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
dfheat = pd.read_csv('ksev1.csv', names=['2Theta','Int'])
df2heat = pd.read_csv('ksev1rand.csv', names=['2Theta','Int2'])
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



#dfheat=dfheat.values
#df['Int2'] = df['Int' + 'Int2']
#ax = df.plot.hist(bins=10, alpha=0.5)
#print(dfheat)
#c1, c2 = st.columns((2,1))
#with c1:
#    st.markdown('### XRD Heatmap')
#    plost.scatter_hist(
#        data=dfheat,
#        x='2Theta',
#        y='Int',
 #       size='Int',
  #      color='Int',
   #     opacity=0.5,
    #    aggregate='count',
      #  width=200,
     #   height=200,
     #   legend='bottom',
    #    use_container_width=True
    #)
    
    
#    plost.xy_hist(
#        data=dfheat,
 #       x='2Theta',
  #      y='Int',
   #     #x_bin=100,
    #    #y_bin='Int2',
     #   use_container_width=True,
    #)
    
    
    #plost.time_hist(
    #data=stocks,
    #date='date',
    #x_unit='Area',
    #y_unit='Area2',
    #color=time_hist_color,
    #aggregate='median',
    #legend=None,
    #height=345,
    #use_container_width=True)
#with c2:
    #st.markdown('### XRDs vs. in %')
#    plost.donut_chart(
 #       data=stocks,
  #      #theta=donut_theta,
   #     theta='Area',
    #    color='Sample',
     #   legend='bottom', 
      #  use_container_width=True)

# Row C
#st.markdown('### XRD Charts')
#st.markdown('##### Main')
#st.line_chart(weather1, x = '2Theta', y = 'Int', height = plot_height)
#st.markdown('##### Comparing')
#st.line_chart(weather2, x = '2Theta', y = 'Int2', height = plot_height)
#st.markdown('##### Main - Comparing')
#st.line_chart(weather3, x = '2Theta', y = 'Diff', height = plot_height)

# Row D
#st.markdown('### Crystal Size and Strain')
#df = pd.read_fwf('myfilesize.txt')

#df.to_csv('myfilesize.csv')


#print(data)

#f=open(df,'rb') # opens file for reading
#reader = csv.reader(f)
#for line in reader:
#    print(line)

#Cryst3 = open('FWHMFirst.txt', 'r')
#Cryst3 = pd.DataFrame({'Theta': Cryst3[:, 0], 'Int': Cryst3[:, 1], 'Scherrer': Cryst3[:,2]})

#df =pd.DataFrame(df)
#del data[data.columns[0]]
#del data[data.columns[0]]
#df.to_csv('FHWMFirstSecond.csv')
#print(data)
#df =pd.DataFrame(df)
#df.columns =['Scherrer']
#print(df)
#del df[df.columns[0]]

#df.to_csv('FHWMFirstSecond2.csv')
#Cryst5 = data['Scherrer'].mean()

#df1 = pd.read_csv('FHWMFirstSecond.csv', names=['Int','Int2' , 'Scherrer'])

#del df1[df1.columns[0]]
#my_column = df1["Scherrer"]
#print(df1)
#df1.pop(df1.columns[0])
#df1 = pd.read_csv(df1, names=['Int','Scherrer'])

#Cryst5 = pd.read_fwf(Cryst4)
#Cryst6 = Cryst5.iloc[:,1:]
#df1.drop(columns=df1.columns[0], axis=1, inplace=True)
#np.savetxt('FHWMFirstSecond.txt', df1, fmt='%s', delimiter=' ')
#df2 = pd.DataFrame(df1, columns=['Scherrer'])

#print(my_column)
#df1.columns=['Scherrer']
#df2.rename(index = {0: 'feature_rank'},{1: 'feature_rank'})
#print(df1)


#for col in df1.columns:
#    if (df1.columns[0]) in col:
#        del df1[col]



#df2 = pd.read_csv('FHWMFirstSecond.txt', names=['Int', 'Scherrer'], index_col=False)
#df2 = df.drop(df.columns[0], axis=1, inplace=True)



#Trans = np.transpose(data)
#df5 = np.array([*dfopen])

#Cryst5 = Cryst3[:, 2].mean()



#print(df1)

#Cryst4 = df.reset_index(drop=True)
#print(Cryst4)



#Cryst['Type'] = ['Crystal Strain', 'Crystal Size after WH' ]
#st.markdown('##### Main')
#st.table(data=Cryst)

#print(Cryst)

#st.markdown('##### Main W-H')
#st.line_chart(weather3, x = 'Theta', y = 'Diff', height = plot_height)
#st.line_chart(weather_WH, x = 'sinTheta', y = 'BetaCosTheta', height = plot_height)

#
#data = xrd.Data('ksev1.xy').importfile()
#charts = xrd.Chart(*data)

#charts.backsub(tol=1,show=False)
#charts.allpeaks(tols=(0.0922,0.8), verbose=False, show=True)
    #Printfunc2 = xrd.Chart.Printfunc(chart)
    #np.savetxt('FWHMfinal.xy', Printfunc2, fmt='%f', delimiter='\t')
#plt.xlabel('2 $\\theta$')
#plt.suptitle('backsub & Automated Scherrer width calculation of all peaks*')
#plt.show()
#uploaded_file = st.image("WH-PLOT.png")

#if uploaded_file is not None:
#    # Convert the file to an opencv image.
#    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
#    st.image(opencv_image, channels="BGR")



#x, y = xrd.Data('ksev1.xy').importfile()
#model = xrd.Rietveld(x, y)
#model.refine()




#uploaded_file2 = st.image("RietveldRef.png")
#uploaded_file = st.file_uploader("Choose Main File")
#dataframe = pd.read_csv(uploaded_file)

#uploaded_file2 = st.image("RietveldRef.png")
#uploaded_file = st.file_uploader("Choose Main File")
#dataframe = pd.read_csv(uploaded_file)


import xrdfit
import lmfit
import pandas as pd
import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt

import dill
import tqdm
#import xrayutilities as xu
import time

from numpy import arange, inf
from matplotlib.pylab import (figure, legend, semilogy, tight_layout, xlabel,
                              ylabel)

import xrayutilities as xu

from xrdfit.spectrum_fitting import PeakParams, FitSpectrum


# Set the matplotlib backend and make the plots a bit bigger

import matplotlib
matplotlib.rcParams['figure.figsize'] = [8, 6]


im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"><b> Detects the most prominent peaks in the range of selected °2Theta values. <b></font> </div>', unsafe_allow_html=True)


values = st.slider(
     'Select a range of °2 Theta values',
     0.0, 180.0, (25.0, 75.0))
#st.write('Values:', values)

#st.write('Values2:', values[1])



st.sidebar.header('')




#mpl.rcParams['font.size'] = 16.0
#mpl.rcParams['lines.linewidth'] = 2.0

#xf = xu.io.XRDMLFile('./pages/KSeV1Rand.xrdml')


#sample = "KSeV1Rand"
#energy = 50000
#center_ch = 700.0002
#chpdeg = 340.0002
#nchannel = 3000
#datapath = os.path.join("KSeV1Rand")






#om, tt, psd = xu.io.getxrdml_map('KSeV1Rand.xrdml', path='data')

#tt, om, psd = xu.io.getxrdml_scan('rsm_%d.xrdml.bz2', 'Omega', scannrs=[1, 2, 3, 4, 5], path='data')

#first_cake_angle = 90

#file_path = pd.read_csv('../ksev1.csv', names=['Theta','Int'])
#spectral_data = FitSpectrum(file_path, first_cake_angle, delimiter=',')





#xf = xu.io.XRDMLFile('data/rsm_1.xrdml.bz2')

#file_path = pd.read_csv('ksev1.csv')
#file_path2 = pd.read_csv('ksev1rand.csv')


first_cake_angle = 120

file_path = "ksev1.csv"
file_path2 = "ksev1rand.csv"

weather1 = pd.read_csv(file_path, names=['\u00b0 2Theta','Int'])

st.markdown('##### Main')
st.line_chart(weather1, x = '\u00b0 2Theta', y = 'Int', height = 250)





print(file_path)

spectral_data = FitSpectrum(file_path, first_cake_angle, delimiter=',')
spectral_data2 = FitSpectrum(file_path2, first_cake_angle, delimiter=',')
print(spectral_data)

spectral_data.plot_polar()

#sd1 = pd.DataFrame(spectral_data)
#sd2 = pd.DataFrame(spectral_data2)
#spectral_data.plot(1)

#spectral_data_MC = sd1 - sd2


spectral_data.plot(1, log_scale=True)
spectral_data2.plot(1, log_scale=True)
#spectral_data_MC.plot(1, log_scale=True)


#spectral_data.plot(1, x_range=(2.7, 50), show_points=True)



peak_params = PeakParams((values[0], values[1]), '(10-10)')


spectral_data.fit_peaks(peak_params, 1)



spectral_data.fitted_peaks[0].result.values

spectral_data.get_fit("(10-10)")


#spectral_data.get_fit("(10-10)").result



#spectral_data.get_fit("(10-10)").result.values



#spectral_data.get_fit("(10-10)").result.values['maximum_0_center']

#spectral_data.get_fit("(10-10)").plot()





#peak_params = [#PeakParams((2.75, 20.95), '1'),
               #PeakParams((20.02, 30.15), '2'),
               #PeakParams((30.15, 40.35), '3'),
#               PeakParams((18, 26), '4')]

#spectral_data.plot_peak_params(peak_params, 1, x_range=(2.7, 60), show_points=True)

#spectral_data.fit_peaks(peak_params, 1)

#print(spectral_data.fitted_peaks)

#for fit in spectral_data.fitted_peaks:
#    fit.plot()





#peak_params = [PeakParams((3.02, 30.27), '(10-10)'),
#               PeakParams((30.3, 50.75), ['(0002)', '(110)', '(10-11)'], [(3.4, 3.44), (3.52, 3.56), (3.57, 3.61)])]

#spectral_data.plot_peak_params(peak_params, 1, show_points=True, label_angle=60)

#spectral_data.plot_fit('(10-10)')
#spectral_data.plot_fit('(0002)')

#st.markdown('###### W-H Plot')
#uploaded_file = st.image("favicon2.png")




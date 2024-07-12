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
import xrdfit.pv_fit

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
st.markdown('<div style="text-align: justify;"> <font size="+3"><b> Detects the most prominent peak in the range of selected °2Theta values. <b></font> </div>', unsafe_allow_html=True)


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

#spectral_data.get_fit("(20-20)").result


spectral_data.get_fit("(10-10)").result


#spectral_data.plot_peak_params

#xrdfit.pv_fit.do_pv_fit(peak_data=spectral_data, peak_param=peak_params)

#spectral_data.plot_fit("(10-10)").__module__.







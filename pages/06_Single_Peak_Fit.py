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

spectral_data.plot(1, log_scale=True)
spectral_data2.plot(1, log_scale=True)

peak_params = PeakParams((values[0], values[1]), '(10-10)')

spectral_data.fit_peaks(peak_params, 1)

spectral_data.fitted_peaks[0].result.values

spectral_data.get_fit("(10-10)")

spectral_data.get_fit("(10-10)").result


xrdfit.plotting.plot_parameter(data=spectral_data, fit_parameter="Int",show_points=True, show_error=True, log_scale=False)

#spectral_data.plot_peak_params

#xrdfit.pv_fit.do_pv_fit(peak_data=spectral_data, peak_param=peak_params)

#spectral_data.plot_fit("(10-10)").__module__.







import xrdfit
import lmfit
import pandas as pd
import numpy as np

# Set the matplotlib backend and make the plots a bit bigger

import matplotlib
matplotlib.rcParams['figure.figsize'] = [8, 6]

from xrdfit.spectrum_fitting import PeakParams, FitSpectrum

import dill
import tqdm


first_cake_angle = 180

#file_path = pd.read_csv('../ksev1.csv', names=['Theta','Int'])
file_path = "ksev1.csv"
file_path2 = "ksev1rand.csv"




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



peak_params = PeakParams((2.75, 50), '(10-10)')


spectral_data.fit_peaks(peak_params, 1)



spectral_data.fitted_peaks[0].result.values

spectral_data.get_fit("(10-10)")


spectral_data.get_fit("(10-10)").result



spectral_data.get_fit("(10-10)").result.values



spectral_data.get_fit("(10-10)").result.values['maximum_0_center']

spectral_data.get_fit("(10-10)").plot()




peak_params = [PeakParams((2.75, 20.95), '1'),
               PeakParams((20.02, 30.15), '2'),
               PeakParams((30.15, 40.35), '3'),
               PeakParams((40.13, 50.30), '4')]

spectral_data.plot_peak_params(peak_params, 1, x_range=(2.7, 60), show_points=True)

spectral_data.fit_peaks(peak_params, 1)

print(spectral_data.fitted_peaks)

for fit in spectral_data.fitted_peaks:
    fit.plot()





peak_params = [PeakParams((3.02, 30.27), '(10-10)'),
               PeakParams((30.3, 50.75), ['(0002)', '(110)', '(10-11)'], [(3.4, 3.44), (3.52, 3.56), (3.57, 3.61)])]

spectral_data.plot_peak_params(peak_params, 1, show_points=True, label_angle=60)

#spectral_data.plot_fit('(10-10)')
#spectral_data.plot_fit('(0002)')




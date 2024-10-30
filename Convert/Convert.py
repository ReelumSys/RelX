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
import urllib

from scipy.optimize import fmin
from scipy.optimize import curve_fit


import contextlib



st.set_page_config(
        page_title="RelX v0.9",
)

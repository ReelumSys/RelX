# Contents of ~/my_app/pages/page_2.py
import streamlit as st
import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
from PIL import Image

im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("../RelX/images/favicon.png");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


#st.markdown("### Acknowledgement")
st.write("###### Thanks to PowerXRDs work by Andrew Garcia Ph.D (https://github.com/andrewrgarcia/powerxrd) a Williamson-Hall-Plot and other functions were written. The Williamson-Hall-Plot is based on the same values as the Scherrer calculation. Furthermore a Halder and Wagner-Plot is made possible from the Literature. The plots should serve as a direct comparison between the three methods.")
st.text("")
st.write("First you can upload two XRD charts. They need not to have the same 2Theta values as you can set a beginning value. However the patterns have to be recorded at same diffraction settings. The main runs Crystal Size and Rietveld Refinement. After uploading, first you see a heatmap plus indentified and binned values. This should help with the data conformity. Beside the heatmaps you can find an analysis of data in the donut chart. As you loaded two charts, the backsubbed area from PowerXRD is read. In the Charts section there are three plots where the last plot is a subtraction between the Main XRD and the Comparison XRD data. After the three, the same procedure is done with in logarithmic scale for the intensity. This should provide an easy access.")
st.text("")
st.write("The Rietveld equation and the parameters were modified and are working! However lmfit does a bad job here, as it seems that the purpose is to fit through multiple points close together at the x-axis. Ongoing work is done here. Need a conversion to a line graph and still run all parameters. Or you are welcome to try with a very long diffraction time.")
st.text("")
st.write("##### Literature")
st.write("Rietveld, H.M. (1969), A profile refinement method for nuclear and magnetic structures. J. Appl. Cryst., 2: 65-71. https://doi.org/10.1107/S0021889869006558")
st.write("W. H. Hall. (1949) Proc. Phys. Soc. A (London, U. K.) 62 741–743")
st.write("Flores-Cano, D. A., Chino-Quispe, A. R., Rueda Vellasmin, R., Ocampo-Anticona, J. A., González, J. C., & Ramos-Guivar, J. A. (2021). Fifty years of Rietveld refinement: Methodology and guidelines in superconductors and functional magnetic nanoadsorbents. Revista De Investigación De Física, 24(3), 39-48. https://doi.org/10.15381/rif.v24i3.21028")
st.write("Izumi, F., Ikeda, T. (2014) - Implementation of the Williamson–Hall and Halder and Wagner Methods into RIETAN-FP")
st.write("The Rietveld Refinement Method: Half of a Century Anniversary Tomče Runčevski and Craig M. Brown Crystal Growth & Design (2021) 21 (9), 4821-4822 DOI: 10.1021/acs.cgd.1c00854")



#st.sidebar.markdown("##### vladimirvopravil@hotmail.com")


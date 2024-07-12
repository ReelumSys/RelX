# Contents of ~/my_app/pages/page_3.py
import streamlit as st
import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
import os
from PIL import Image
from streamlit_extras.app_logo import add_logo

im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    #page_icon=im,
    layout="wide",
)
#add_logo("favicon3.png")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

#st.markdown("## Imprint")
st.sidebar.markdown("# ")
st.write("###### \n Impressum (Legal Notice) ff") 
st.write('''"Information according to § 5 TMG: "''') 
st.write("Vladimir Vopravil \n\r",
         "30161 Hannover \n\r",
         "vladimirvopravil@hotmail.com \n\r",
         "#### Disclaimer \n\r",
         "###### Liability for Content \n\r",
         "The content of our pages has been created with the utmost care. However, we cannot guarantee the accuracy, completeness, or timeliness of the content. As a service provider, we are responsible for our own content on these pages according to § 7 (1) TMG. According to §§ 8 to 10 TMG, we are not obligated to monitor transmitted or stored third-party information or to investigate circumstances that indicate illegal activity. Obligations to remove or block the use of information under general laws remain unaffected. However, liability in this regard is only possible from the moment of knowledge of a specific infringement. Upon becoming aware of such infringements, we will remove the content immediately. \n\r",
         "###### Liability for Links \n\r",
         "Our website contains links to external third-party websites over which we have no control. Therefore, we cannot assume any liability for these external contents. The respective provider or operator of the linked pages is always responsible for their content. The linked pages were checked for possible legal violations at the time of linking. Illegal content was not recognizable at the time of linking. However, a permanent control of the linked pages is not reasonable without concrete evidence of a violation of the law. Upon becoming aware of any legal infringements, we will remove such links immediately. \n\r",
         "###### Copyright \n\r",
         "The content and works created by the site operators on these pages are subject to German copyright law. Duplication, editing, distribution, and any kind of use outside the limits of copyright law require the written consent of the respective author or creator. Downloads and copies of this site are only permitted for private, non-commercial use. Insofar as the content on this site was not created by the operator, the copyrights of third parties are respected. In particular, third-party content is identified as such. Should you become aware of a copyright infringement, please inform us accordingly. Upon becoming aware of any infringements, we will remove such content immediately. \n\r",
         "###### Data Protection \n\r",
         "The use of our website is usually possible without providing personal information. If personal data (such as name, address, or email addresses) is collected on our pages, it is always done on a voluntary basis, as far as possible. This data will not be disclosed to third parties without your explicit consent. We would like to point out that data transmission over the internet (e.g., communication by email) can have security vulnerabilities. A complete protection of the data against access by third parties is not possible. The use of contact data published within the framework of the imprint obligation by third parties for sending unsolicited advertising and information materials is hereby expressly prohibited. The operators of the pages expressly reserve the right to take legal action in the event of the unsolicited sending of advertising information, such as spam emails. \n\r",
         "Impressum supported by Kanzlei Hasselbach."
            

         )













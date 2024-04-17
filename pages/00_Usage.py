import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
from PIL import Image
import streamlit as st
import random
import time
import openai
import os
from langchain.llms import OpenAI
#from streamlit_extras.app_logo import add_logo
from utils import logo
from streamlit_extras.app_logo import add_logo

im = 'favicon2.png'
st.set_page_config(
    
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)
#logo()

add_logo("favicon3.png")
#st.sidebar.image("./images/favicon.png", width=150)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("../images/favicon.png");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"><b> With this app it is possible to do some work on XRD patterns with only minimal amounts of data needed. For the Symmetry depicition HKL values are required. Hence for Rietveld Refinement also Atomic Coordinates and Atomic Dislocation are mandatory. <b></font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"> First you can upload two XRD charts. They need not to have the same °2Theta values as you can set a beginning value. However the patterns have to be recorded at same diffraction settings. After uploading, first you see a heatmap plus identified and binned values. This should help with the data conformity. Beside the heatmap you can find an analysis of data in the donut chart. Here is the Main vs. the Comparing pattern plottet. In the pattern section is an overview of the plotted patterns. A subtraction from the Main XRD and the Comparison XRD is shown. The same procedure is done with logarithmic scale for the intensity. Publication ready plots may be included with easy access. </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"> The main runs Crystal Size and Rietveld Refinement. The Rietveld equation and the parameters were modified, only few additional values need to be set like the position of the atoms, or scale factor. These values need to be incorporated on the front-end. Ongoing work is done here. Maybe heading for a conversion to a line graph and still run all parameters. Or you are welcome to try with a very long diffraction time. </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"> A Discord server has been set up for possible further development. <a href="https://discord.gg/gFjuBQd4">Click here</a> </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3">Below is an Chatbot, where you can post questions about XRD and life. It is based on OpenAIs work. Soon it will have knowledge about this programm. </font> </div>', unsafe_allow_html=True)


#st.title("XRDGPT")
#OPENAI_API_KEY = "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"
#openai.api_key = st.secrets["OPENAI_API_KEY"]
#openai.api_key = st.secrets["OPENAI_API_KEY"]


#if "openai_model" not in st.session_state:
#    st.session_state["openai_model"] = "gpt-3.5-turbo"

#if "messages" not in st.session_state:
#    st.session_state.messages = []

#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])

#if prompt := st.chat_input("What is up?"):
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    with st.chat_message("user"):
#        st.markdown(prompt)

#    with st.chat_message("assistant"):
#        message_placeholder = st.empty()
#        full_response = ""
#        for response in openai.ChatCompletion.create(
#            model=st.session_state["openai_model"],
#            messages=[
#                {"role": m["role"], "content": m["content"]}
#                for m in st.session_state.messages
#            ],
#            stream=True,
#        ):
#            full_response += response.choices[0].delta.get("content", "")
#            message_placeholder.markdown(full_response + "▌")
#        message_placeholder.markdown(full_response)
#    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.title('XRDGPT')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to do a Rietveld Refinement?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)


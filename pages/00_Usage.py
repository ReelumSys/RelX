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
from langchain.llms.openai import OpenAI


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


st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"><b> With this app it is possible to do some work on XRD Charts with only minimal amounts of data needed. For the Bravais depiciton only HKL values are needed. <b></font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"> First you can upload two XRD charts. They need not to have the same ° 2Theta values as you can set a beginning value. However the patterns have to be recorded at same diffraction settings. The main runs Crystal Size and Rietveld Refinement. After uploading, first you see a heatmap plus identified and binned values. This should help with the data conformity. Beside the heatmaps you can find an analysis of data in the donut chart. Here is the Main vs. the Comparing diffractogramm plottet. As you loaded two charts, the backsubbed area from PowerXRD is read. In the Charts section there are three plots where the last plot is a subtraction between the Main XRD and the Comparison XRD data. After the three, the same procedure is done with in logarithmic scale for the intensity. This should provide an easy access with publication ready plots also in other functions. </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3">The Rietveld equation and the parameters were modified and are working. However it seems that lmfit has a purpose to fit through points closer together at the x-axis. Ongoing work is done here. Needing a conversion to a line graph and still run all parameters. Or you are welcome to try with a very long diffraction time. </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"> A Discord server has been set up for possible further development. <a href="https://discord.gg/gFjuBQd4">Click here</a> </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3">Below is an XRD Chatbot, where you can post questions about XRD and life. It is based on openAIs work. Soon it will have knowledge about this programm. </font> </div>', unsafe_allow_html=True)


st.title("XRDGPT")
#OPENAI_API_KEY = "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"
#openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.title('🦜🔗 Quickstart App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)


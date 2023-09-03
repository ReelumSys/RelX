# Contents of ~/my_app/pages/page_0.py
import streamlit as st
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
st.write("###### With this app it is possible to do some work on XRD Charts with only minimal amounts of data needed. Since for Rietveld Refinement only atomic parameters and HKL-values are needed. Same for the Bravais crystal structure.")
st.text("")
st.write("###### The Charts section is also for comparing data. The Main gets subtracted by the Comparing. The same is done with logarithmic values on the y-axis.")





st.title("XRDGPT")
#OPENAI_API_KEY = "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"
#openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5"

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
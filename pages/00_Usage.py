import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
from PIL import Image
import random
import time
#import openai
import os

import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from langchain.llms import OpenAI
#from langchain_openai import OpenAI



#from flask import Flask, request, jsonify
#from flask import Flask, render_template, request, url_for



import sys

from streamlit_chat import message
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma

import warnings
warnings.filterwarnings("ignore")

#from llama_index.core import SimpleDirectoryReader, GPTListIndex, PromptHelper, KnowledgeGraphIndex, LLMPredictor
#from llama_index import LLMPredictor
#from langchain.chat_models import ChatOpenAI
#import gradio as gr
#import sys
#import os
#import time
#import scikit-learn
#import scipy
#import openai
#from openai.embeddings_utils import get_embedding, cosine_similarity
#import pandas
#import openai
#import numpy as np
#import glob
#import datetime



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
st.markdown('<div style="text-align: justify;"> <font size="+6"><b> Introduction <b></font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"><b> With this app it is possible to do some work on XRD patterns with only minimal amounts of data needed. For the Symmetry depicition HKL values are required. Hence for Rietveld Refinement also Atomic Coordinates and Atomic Dislocation are mandatory. <b></font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"> First you can upload two XRD charts. They need not to have the same °2Theta values as you can set a beginning value. However the patterns have to be recorded at same diffraction settings. After uploading, first you see a heatmap plus identified and binned values. This should help with the data conformity. Beside the heatmap you can find an analysis of data in the donut chart. Here is the Main vs. the Comparing pattern plottet. In the pattern section is an overview of the plotted patterns. A subtraction from the Main XRD and the Comparison XRD is shown. The same procedure is done for the intensity in logarithmic scale. Publication ready plots may be included with easy access. </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"> The main runs Crystal Size and Rietveld Refinement. The Rietveld equation and the parameters were modified, only few additional values need to be set like the position of the atoms, or scale factor. These values need to be incorporated on the front-end. Ongoing work is done here. Maybe heading for a conversion to a line graph and still run all parameters. Or you are welcome to try with a very long diffraction time. </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3"> A Discord server has been set up for possible further development. <a href="https://discord.gg/gFjuBQd4">Click here</a> </font> </div>', unsafe_allow_html=True)

st.text("")
st.markdown('<div style="text-align: justify;"> <font size="+3">In the Chatbot below, you can post questions about XRD and life. It is based on OpenAIs work. Soon the Chatbot will have knowledge about this programm. </font> </div>', unsafe_allow_html=True)


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
    llm = OpenAI(temperature=0.01, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to do a Rietveld Refinement?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)





def generate_response(txt):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)


# Page title
#st.set_page_config(page_title='Text Summarization App')
st.title('Text Summarization App')


# Text input
txt_input = st.text_area('Enter your text', '', height=200)
# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type = 'password', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
            del openai_api_key
if len(result):
    st.info(response)










st.title("XRDGPT - Personal Assitant")
st.divider()

data_file = "./pages/data.txt"
data_persist = False
prompt = None

#containers for the chat 
request_container = st.container()
response_container = st.container()

# Persist and save data to disk using Chroma 
if data_persist and os.path.exists("persist"):
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader(data_file)
    loader.load()
    if data_persist:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}))

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! I am your Personal assistant built by XYZ Books"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

def conversational_chat(prompt):
    result = chain({"question": prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((prompt, result["answer"]))
    return result["answer"]


with request_container:
    with st.form(key='xyz_form', clear_on_submit=True):
        
        user_input = st.text_input("Prompt:", placeholder="Message XYZBot...", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer", seed=13)
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed=2)
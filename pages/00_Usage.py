


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



from utils import logo
from streamlit_extras.app_logo import add_logo

im = 'favicon2.png'
st.set_page_config(
    
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)
#logo()

#add_logo("favicon3.png")
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
st.markdown('<div style="text-align: justify;"> <font size="+3"><b> With this app it is possible to do some work on XRD patterns with only minimal amounts of data needed. For the Symmetry depicition HKL values are required. <b></font> </div>', unsafe_allow_html=True)




st.title('GPT')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    llm = OpenAI(temperature=0.01, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', '')
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












st.title("GPT - Personal Assitant")
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
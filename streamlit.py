from re import L
import streamlit as st
# https://github.com/AI-Yash/st-chat
from streamlit_chat import message
import pandas as pd
import numpy as np
from returnreply import returnreply

st.set_page_config(
    page_title="Epic Amazing Chatbot" 
)

st.header("Streamlit Chat - Demo")

def on_change():
    message(returnreply(user_input), is_user=False)

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input", on_change=on_change)
    return input_text 

user_input = get_text()

message("My message") 
message("Hello bot!", is_user=True)  # align's the message to the right

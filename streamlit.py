from re import L
import streamlit as st
# https://github.com/AI-Yash/st-chat
from streamlit_chat import message
from returnreply import returnreply

input = st.text_input('Your Message') #awdad

if input:
    message("You: " + input, is_user=True)
    message(returnreply(input), is_user=False)
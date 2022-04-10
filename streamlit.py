import streamlit as st
import pandas as pd
import numpy as np
from returnreply import returnreply

st.title("Chatbot!!!!!!")

messages = []

def clear_text():
    messages.append(input)
    for word in messages:
        st.write(word)
    st.session_state["textinput"] = ""

input = st.text_input("Your Input:", on_change=clear_text, key="textinput")
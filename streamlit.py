from re import L
import streamlit as st
# https://github.com/AI-Yash/st-chat
from streamlit_chat import message
import pandas as pd
import numpy as np
from returnreply import returnreply

#st.write("Welcome to the Chatbot. I am still learning, please be patient")
#input = st.text_input('User:')

# TEXT INPUT BOX
# - when enter press: clear, do do function onInput()
# onInput() FUNCTION
# - write the input as a user message
# - write the bot reply



#message("Hi Bot", is_user=True, key="message1")
#message("I am a chatbot", key="message2")
#message("Do you take paypal?", is_user=True, key="message3")
#message("Yes we do lol", key="message4")

input = st.text_input('Your Message')

#st.write('You: ', input)
if input:
    message("You: " + input, is_user=True)
    message("Jonathan (Bot): " + returnreply(input), is_user=False)
#st.write('Jonathan (Bot): ', returnreply(input))
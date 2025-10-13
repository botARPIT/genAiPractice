##Simple web app for querying ai model

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

st.header('Document summarizer')
user_input = st.text_input('Enter your prompt')
response = model.invoke(user_input)
if st.button('Summarize'):
    st.write(response.content)

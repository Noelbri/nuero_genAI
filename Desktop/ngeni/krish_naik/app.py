import os
from apikey import apikey
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ["GROQ_API_KEY"]= apikey
st.title('Medium Article Generator')
topic = st.text_input('Input your topic of interest')
language = st.text_input('Input language')
title_template = PromptTemplate(
    input_variables = ['topic', 'language'],
    template='Give me medium article title on {topic} in {language}'
)
article_template = PromptTemplate(
    input_variables = ['title']
    template = 'Give me medium article title for {title}'
)
#To establish a connection with OpenAI, we first instantiate an OpenAI instance. Within this constructor, we specify the temperature, which I'll set to 0.9.
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
article_chain = LLMChain(llm=llm, prompt=article_template, ver)
#If a topic is provided, it will be processed by the llm and the response will be displayed in the user interface.
if topic:
    response = title_chain.run({'topic':topic,'language':language})
    #response = llm([HumanMessage(content=prompt)])
    st.write(response)
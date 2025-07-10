import os
from constants import openai_api_key
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Streamlit app title
st.title("TV MODELS RESPONSE CONNECTION")
input_text = st.text_input("Enter your tv model:")

# Initialize the OpenAI LLM
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8)

# Custom prompt template
first_prompt_template = PromptTemplate(
    input_variables=["model"],
    template="give me info about {model} tv model"
)

# Create the LLM chain
chain1 = LLMChain(llm=llm, prompt=first_prompt_template,verbose=True,output_key="info")

#second prompt template
second_prompt_template = PromptTemplate(
    input_variables=["info"],
    template="In which year was {info} released?"
)
# Create the second LLM chain
chain2 = LLMChain(llm=llm, prompt=second_prompt_template,verbose=True,output_key="year")

#third prompt template
third_prompt_template = PromptTemplate(
    input_variables=["year"],
    template="5 Major events occur in {year}"
)
# Create the third LLM chain
chain3 = LLMChain(llm=llm, prompt=third_prompt_template,verbose=True,output_key="events")

parent_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["model"],
    output_variables=["info", "year", "events"],
    verbose=True
)

# Run the chains with the input text
if input_text:
    st.write(parent_chain.invoke({"model": input_text}))
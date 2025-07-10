import os
from constants import openai_api_key
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

#streamlit app title
st.title("Enter Company Name")
input_text = st.text_input("Enter your text here:")

# custom prompt template
first_prompt_template = PromptTemplate(
    input_variables=["company"],
    template="give me information about {company} company"
)

# Initialize the OpenAI LLM
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8)
# Create the LLM chain
chain = LLMChain(llm=llm, prompt=first_prompt_template, verbose=True)

# Run the chain with the input text
if input_text:
    response = chain.run(company=input_text)
    st.write(response)

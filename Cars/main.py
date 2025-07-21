from langchain_community.llms import CTransformers
import streamlit as st
from langchain.prompts import PromptTemplate

def output(company, year, no):
    llm=CTransformers(model="O:\LANGCHAIN\Cars\models\llama-2-7b-chat.ggmlv3.q8_0.bin", #need model offline
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})

                        
    template = """ List {no} of cars manufactured by {company} in {year}."""
    prompt = PromptTemplate(
        template=template,
        input_variables=["company", "year", "no"]
    )
    response = llm.invoke(prompt.format(company=company, year=year, no=no))
    return response

# streamlit app title
st.set_page_config(page_title="Search Cars",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Search Cars 🤖")

company = st.text_input("Enter the company name:")
col1, col2 = st.columns([5, 5])

with col1:
    year = st.text_input('enter year of manufacture:')
with col2:
    no = st.selectbox('how many cars do you want to see?',
                      ('3', '5', '10'), index=0)

submit = st.button("Generate")

if submit:
    st.write(output(company, year, no))

import os
from constants import openai_api_key
import pypdf as pdf
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
# from typing_extensions import concatenate

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Streamlit app title
st.title("QNA BOT")
input_text = st.text_input("Enter your question related to the document:")

# Load the PDF document
pdf_document = pdf.PdfReader("document.pdf")

# Extract text from the PDF
pdf_text = ""
for page in pdf_document.pages:
    pdf_text += page.extract_text()

# Split the text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_text(pdf_text)

# Create embeddings for the text chunks
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

# Create a question-answering chain
qa_chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Process user input
if input_text:
    docs = docsearch.similarity_search(input_text)
    answer = qa_chain.run(input_documents=docs, question=input_text)
    st.write(answer)
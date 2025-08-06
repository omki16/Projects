from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import os


loader = PyPDFLoader("retrival/budget.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,                  #not better results due to this splitter
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama2") #less accurate results with this model, try with other models like llama3, gemini etc.

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Create (or load) persistent FAISS vectorstore
vector = embeddings.embed_query("test sentence")
print("Embedding dimension:", len(vector))

dimension = len(vector)
index = faiss.IndexFlatL2(dimension)

vector_store = FAISS(
    embedding_function=embeddings,
    docstore=InMemoryDocstore(),
    index=index,
    index_to_docstore_id={},
)

print("Adding documents to FAISS...")
vector_store.add_documents(chunks)

query = "what does The great Telugu poet and playwright Gurajada Appa Rao had said?"
results = vector_store.similarity_search_with_score(query, k=3)

print("\nResults:")
print(results)

from langchain_core.prompts import ChatPromptTemplate
# prompt making
prompt = ChatPromptTemplate.from_template(
    "You are qna chatbot Answer each and every the question based on the provided context only else say you dont know: {context}\nQuestion: {input}\nAnswer:")

llm = OllamaLLM(model="llama2")

from langchain.chains.combine_documents import create_stuff_documents_chain
#chain creation
chain = create_stuff_documents_chain(llm, prompt)

# making retrival and retrival chain
retrieval = vector_store.as_retriever()

from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retrieval, chain)

#retrival invoking

response = retrieval_chain.invoke({
    "input": "what does The great Telugu poet and playwright Gurajada Appa Rao had said?",
})
print(response['answer'])
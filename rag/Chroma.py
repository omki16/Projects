from langchain_community.document_loaders import TextLoader, WebBaseLoader
from bs4 import SoupStrainer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# Set user-agent to avoid warning
os.environ["USER_AGENT"] = "LangChainBot/1.0"

# Load local text file
loader = TextLoader(
    r"O:\PERSONAL\LANGCHAIN UPDATED\rag\hello.txt", 
    encoding="utf-8"
)
text_docs = loader.load()



# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
text_chunks = text_splitter.split_documents(text_docs)
web_chunks = text_splitter.split_documents(web_docs)

# Display sample chunks for debugging
print("Sample chunk:")
for chunk in text_chunks[:1]:
    print(chunk.page_content[:300])
print("\n---\n")

# Set up embeddings
embeddings = OllamaEmbeddings(model="llama2")

# Create (or load) persistent Chroma vectorstore
persist_dir = "./chroma_store"
vectorstore = Chroma(
    collection_name="text_chunks",
    embedding_function=embeddings,
    persist_directory=persist_dir
)

# Only add documents if Chroma is empty
if len(vectorstore.get()['documents']) == 0:
    print("Adding documents to ChromaDB...")
    vectorstore.add_documents(text_chunks + web_chunks)
    vectorstore.persist()
else:
    print("ChromaDB already populated.")

# Perform similarity search
query = "Beirut explosion ammonium nitrate 2020"
results = vectorstore.similarity_search_with_score(query, k=3)

print("\nResults:")
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(doc.page_content[:300], "\n---")

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from bs4 import SoupStrainer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import os

# Set user-agent to avoid warning
os.environ["USER_AGENT"] = "LangChainBot/1.0"

# Load webpage
Webloader = WebBaseLoader(
    web_paths=["https://python.langchain.com/docs/how_to/document_loader_web/"],
    bs_kwargs={
        "parse_only": SoupStrainer(class_="language-output codeBlockContainer_Ckt0 theme-code-block"),
    },
    bs_get_text_kwargs={"separator": " | ", "strip": True},
)
web_docs = Webloader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

web_chunks = text_splitter.split_documents(web_docs)

embeddings = OllamaEmbeddings(model="llama2")

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

# Only add documents if FAISS is empty

print("Adding documents to FAISS...")
vector_store.add_documents(web_chunks)

# Perform similarity search
query = "what is webbasedloader?"
results = vector_store.similarity_search_with_score(query, k=3)

print("\nResults:")
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(doc.page_content[:300], "\n---")

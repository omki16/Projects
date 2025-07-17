import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 3. Pinecone setup
api_key = "pcsk_2pQvuS_5TB7HhEJMcgK3UTZgAGnE2CThwA6kD6qauHitqHPJewEM5tX1FDsKyFSVLnZx6R"
os.environ["PINECONE_API_KEY"] = api_key
pc = Pinecone(api_key=api_key)

# 2. Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # no need of api key i locally downloaded in my env 
test_vector = embedding_model.embed_query("sample text")
dimension = len(test_vector)

# 1. Load and split documents
loader = PyPDFDirectoryLoader("O:/LANGCHAIN/vectoruse")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
split_docs = splitter.split_documents(documents)



index_name = "docs-example"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# 4. Create the LangChain vectorstore using the official `langchain_pinecone`
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,                                                      #new version of storing embeddings
    text_key="text",
)

# Add documents
vectorstore.add_documents(split_docs)


from langchain_huggingface import HuggingFaceEndpoint
from constants import HF_TOKEN
from langchain.chains.question_answering import load_qa_chain

import os 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

repo_id = "mistralai/Devstral-Small-2507"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    provider="auto", 
)

def query_index(query):
    # Query the Pinecone index
    results = vectorstore.similarity_search(query, k=5)
    return results

def answer_question(question):
    # Answer a question using the QA chain
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    results = query_index(question)
    answer = chain.invoke(input_documents=results, question=question)
    return answer

print( answer_question("what is capital of france?"))



from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama.llms import OllamaLLM
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

import faiss

# Load Wikipedia tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Load PDF and split into chunks
loader = PyPDFLoader("retrival/parts.pdf")
docs = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)

chunks = text_splitter.split_documents(docs)

# Create embeddings
embeddings = OllamaEmbeddings(model="llama2")#use other models llama2 is not made for embeddings
# Less accurate results with this model, try with other models like llama3, gemini etc
vector = embeddings.embed_query("test sentence")
print("Embedding dimension:", len(vector))

# Create FAISS vectorstore
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

query = "John never __________puts sugar in his coffee"
results = vector_store.similarity_search_with_score(query, k=3)
print("\nResults:")
print(results)

# Convert vectorstore to retriever
retrieval = vector_store.as_retriever()

# Create retriever tool
retrieval_tool = create_retriever_tool(
    retriever=retrieval,
    name="pdf_retriever",
    description="Use this to answer questions from the uploaded PDF document."
)


# Load built-in tools
builtin_tools = load_tools(["arxiv"])

# Combine all tools
tools = [retrieval_tool]

# Load prompt and LLM
prompt = hub.pull("hwchase17/react-json")
llm = OllamaLLM(model="llama2")

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create executor with full tool list ✅
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Run the agent
response = agent_executor.invoke(
    {
        "input": "John never __________puts sugar in his coffee."
    }
)

print(response)


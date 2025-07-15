
from langchain_huggingface import HuggingFaceEndpoint
from constants import HF_TOKEN

import os 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

repo_id = "mistralai/Devstral-Small-2507"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    provider="auto", 
)

print(llm.invoke("What is machine learning?"))
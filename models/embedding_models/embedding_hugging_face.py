from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding =  HuggingFaceEmbeddings( model_name = "sentence-transformers/all-MiniLM-L6-v2")

document = [
    "This is a document",
    "Second document"
]
vector = embedding.embed_documents(document)
print(vector)
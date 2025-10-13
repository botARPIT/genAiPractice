from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "NASA has announced the successful launch of the Artemis I mission...",
    "LangChain is a framework for developing applications with large language models...",
    "I recently purchased the Acer Aspire 7 laptop. The performance is solid...",
    "The sun dipped below the horizon, painting the sky in shades of orange and purple...",
    "Q: How do I reset my password if I forget it? A: Click on the 'Forgot Password' link..."
]



query = "Tell me about laptop's performance"

document_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

##Printing similarity score of query with each document
scores = cosine_similarity([query_embeddings], document_embeddings)[0]



similar_embedding = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]
print(f"The given query: {query} => points to the: ")
print(documents[similar_embedding[0]])

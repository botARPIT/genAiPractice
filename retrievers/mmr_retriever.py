# Its an information retrieval algorithm that not only focuses on retrieveing relevant docs but also eliminating any repeated results
# from langchain_community.retrievers import ma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

sample = [
    Document(page_content= "Postgres DB is the new db of the model era with highly performant engine"),
    Document( page_content = "The Rust programming language has become the go to choice of most devs for building ultra low latency systems"),
    Document( page_content = "Go has also become the choice of most startupts for building fast and reliable backend systems not fast as Rust though"),
    Document( page_content = "Go is a popular choice for building fast and reliable backend systems not fast as Rust though"),
    Document(page_content= "Java on the other hand has been the choice of enterprises due to it large codebase")
]

# Creating vector store
vector_store = FAISS.from_documents(
    documents = sample,
    embedding = embedding_model
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = { "k" : 3, "lambda_mult" : 0.5} # lambda_mult helps the algorithm to give diversified result, its value ranges from 0 to 1, the less the value more the diversified result
)

query = "Language to build performant web apps"

result = retriever.invoke(query)

print(result)
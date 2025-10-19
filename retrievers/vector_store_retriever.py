from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

embedding_model = HuggingFaceEmbeddings( model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(page_content= "Postgres DB is the new db of the model era with highly perfomant engine"),
    Document( page_content = "The Rust programming language has become the go to choice of most devs for building ultra low latency systems"),
    Document( page_content = "Go has also become the choice of most startupts for building fast and reliable backend systems not fast as Rust though"),
]

vector_store = Chroma.from_documents(
    documents = documents,
    embedding = embedding_model,
    collection_name = "testing_collection"
)

# Converting the vector store into retriever
retriever = vector_store.as_retriever(search_kwargs = {"k" : 2})

query = "Language for building fast systems"

result = retriever.invoke(query)

print(result)

for relevant_doc in result:
    print(relevant_doc.page_content)



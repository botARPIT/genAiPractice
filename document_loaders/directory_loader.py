from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

loader = DirectoryLoader(
    path = "document_loaders/docs",
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.lazy_load() # Used to load only the require document not all at once

for document in docs:
    print(document.page_content)
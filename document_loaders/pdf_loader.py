# It uses pyPdf Library under the hood to load the pdf, but its not good to load scanned pdf or complex pages

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

loader = PyPDFLoader('document_loaders/operation.pdf')

prompt = PromptTemplate(
    template="Summarize the content of the file in 4-5 lines \n {content}",
    input_variables = ['content']
)

parser = StrOutputParser()

docs = loader.load()
print(docs)

chain = prompt | model | parser
response = chain.invoke({'content': docs[0].page_content})
print(response)
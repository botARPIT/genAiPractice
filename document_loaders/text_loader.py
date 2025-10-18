# This loader is used to extract text from .txt files

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

loader = TextLoader('document_loaders/test.txt', encoding = 'utf-8')

prompt = PromptTemplate(
    template = "Add more lines to the follwing content \n {content}",
    input_variables = ['content']
)

parser = StrOutputParser()

docs = loader.load()
print(type(docs))
print(docs)
print(len(docs))
print(docs[0].page_content)

chain = prompt | model | parser
response = chain.invoke({'content' : docs[0].page_content})
print(response)

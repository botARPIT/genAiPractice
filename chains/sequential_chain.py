from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import Runnable
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

template = PromptTemplate(
    template = "Generate a detailed report of 200 words on the following topic \n {topic}",
    input_variables=['topic']
)

template1 = PromptTemplate(
    template = "Generate a 5 point summary of the following report \n {report}",
    input_variables=['report']
)

parser = StrOutputParser()

chain = template | model | parser | template1 | model | parser
response = chain.invoke({'topic': 'Communism vs democracy in China'})
print(response)
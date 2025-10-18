# Used to load csv one row at a time
from langchain_community.document_loaders import CSVLoader
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

prompt = PromptTemplate(
    template = "Answer the following question {question} from the following csv data \n {data}",
    input_variables = ['question', 'data']
)

parser = StrOutputParser()

loader = CSVLoader(file_path='document_loaders/backend_api_test_data.csv')

doc = loader.load()

chain = prompt | model | parser
response = chain.invoke({'question': "Which type of request is this?", 'data' : doc[20]})
print(response)
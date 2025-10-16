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

template = PromptTemplate(
    template = "Write 4 lines about the following topic \n {topic}",
    input_variables=["topic"] 
)

parser = StrOutputParser()
chain = template | model | parser
response = chain.invoke({"Democracy in china"})
print(response)
chain.get_graph().print_ascii()
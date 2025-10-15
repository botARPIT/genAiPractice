# Here the output of the model is generated using the default way i.e. using result.content

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

template = PromptTemplate(
    template= "You are expert summary writer, write a 5 line summary on the following topic \n {topic}",
    input_variables=['topic']
)

prompt = template.invoke({'topic': "What is symmetric and assymetric encryption"})
response = model.invoke(prompt)

prompt1 = template.invoke({'topic': response.content})
response1 = model.invoke(prompt1)
print(response1.content)
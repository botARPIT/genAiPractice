# In this file, the output of the llm is parsed using the string output parser in langchain

# The main benefit of using output parser is they can be effectively be integrated with chains in langchain

# Demo workflow: Give a topic to llm => generates a detailed report => give detailed report to llm => llm generates a short summary
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

template = PromptTemplate(
    template="Prepare a detailed report on the following topic: {topic}",
    input_variables=['topic']
)

template1 = PromptTemplate(
    template="Summarize the contents in 3-4 lines: {Content}",
    input_variables=['content']
)

prompt = template.invoke({"topic": "What are agents in agentic ai"})
response = model.invoke(prompt)
prompt1 = template1.invoke({"content": response.content})
response1 = model.invoke(prompt1)
print(response)
print(response1)
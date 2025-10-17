from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel
load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)


llm2 = HuggingFaceEndpoint(
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm = llm2)

template1 = PromptTemplate(
    template= "Generate a short note on the following topic \n {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Generate 5 short quizes with answer on the following topic \n {topic}",
    input_variables=['topic']
)

template3 = PromptTemplate(
    template = "Merge the provided note and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : template1 | model1 | parser,
    'quiz' : template2 | model2 | parser
})

merge_chain = template3 | model1 | parser

chain = parallel_chain | merge_chain

response = chain.invoke({'topic': "Gradient descent"})
print(response)

chain.get_graph().print_ascii()
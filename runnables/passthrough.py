# This is an example of passthrough runnable

from langchain.schema.runnable import RunnablePassthrough, RunnableSequence, RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = "Write a joke on the following topic \n {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Explain the following joke \n {joke}",
    input_variables = ['joke']
)
parser = StrOutputParser()
passthrough = RunnablePassthrough()
chain1 = RunnableSequence(prompt, model, parser)
parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(chain1, parallel_chain)
result = final_chain.invoke({"topic" : "The solar system"})
print(result)

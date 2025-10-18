# In this file, we are using the primitive Runnable Lambda to count the number of words in the response produced by the LLM
# Runnable lambda basically used to add custom logic inside the chain

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = "Generate a joke on the following topic \n {topic}",
    input_variables = ['topic'] 
)

parser = StrOutputParser()

def wordCounter(sentence: str):
    words = sentence.split()
    # print(words)
    return len(words)

joke_gen_chain = RunnableSequence(prompt, model, parser)
# joke_gen_chain = prompt | model | parser # Langchain expressive language (LCEL)
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    # "words" : RunnableLambda(wordCounter),
    'words' : RunnableLambda(lambda x : len(x.split())) # If you want to directly write the logic here
})
# print(joke_gen_chain.invoke({"topic" : "Tourism"}))
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({"topic" : "Fashion industry"})
print(result)

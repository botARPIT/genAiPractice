# Conditonal chains are used to run a particular chain when a particular condition is met
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

parser1 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables = ["feedback"], 
    partial_variables={'format_instruction': parser1.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Write a appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write a appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)


classifier_chain = prompt1 | model | parser1
# response = classifier_chain.invoke({"feedback": "Loved the new scroll animation"})
# print(response)

# Used to excute chain based on condition
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda( lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback': "The app is too buggy and takes years to load images"})
print(result)
chain.get_graph().print_ascii()

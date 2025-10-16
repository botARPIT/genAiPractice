# This parser converts/forces model to produce output in json format
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()
template = PromptTemplate(
    template= "Generate a fiction story of 4-5 lines of a fictional_character with name: {character_name} \n {format_instruction}",
    input_variables= ['character_name'],
    partial_variables={'format_instruction': parser.get_format_instructions()} #Parser defines the output format to llm
)

# # prompt = template.format()
# prompt = template.invoke({'character_name': 'Houdini'})
# print(prompt)


# response = model.invoke(prompt)
# print(response) 
# response_in_json = parser.parse(response.content)
# print(response_in_json)


# using chain
chain = template | model | parser
response = chain.invoke({'character_name': 'Tester'})
print(response)

# The biggest downside of using jsonOutputParser is that u cannot enforce json schema on the model, the model decides the structure of json schema
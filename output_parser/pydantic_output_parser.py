# Used to enfore schema validation on the llm response
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

# This class will act as a schema for our pydantic output parser
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt = 18, description="Age of the person in integer")
    city: str = Field(description="Name of the city where person currently lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Generate fictional name, age, city of a {place} person \n {format_instruction}",
    input_variables=['palce'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({"place": "Turkish"})

result = model.invoke(prompt)
print(result.content)
parsed_result = parser.parse(result.content)
print(parsed_result)


# Using chain
chain = template | model | parser
response = chain.invoke({'place': 'British'})
print(response)


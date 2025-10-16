# This is used to enforce the model (llm) to return response into predefined JSON schema
# The biggest disadvantage of the structured output parser is that we cannot perform data validation on it
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)
 
template = PromptTemplate(
    template = "List 2 facts about blackhole \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({})
response = model.invoke(prompt)

print(response.content)
parsed_output = parser.parse(response.content)
print(parsed_output)


# using chain
chain = template | model | parser
result = chain.invoke({})
print(result)

# This is anothe way of creating a custom tool in langchain, the benefit of using this approach is that unlike type hinting used in tool decorator, here pydantic is used to enforce strict input validations

from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(json_schema_extra={"required" :True, "description" : "First number to add"}),
    b: int = Field(json_schema_extra={"required" :True, "description" : "Second number to add"})

def multiply_fn(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func= multiply_fn,
    name= "Multiply",
    description= "Tool to multiply two numbers",
    args_schema= MultiplyInput
)

print(multiply_tool.invoke({"a" : 2, "b" : 3}))
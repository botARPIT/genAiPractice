# This is another way to create tools in class using the BaseTool class, and the advantage of using this method is, we can create async tools using this unlike the other two approaches, this method provides in-depth customization

# Langchain toolkit is basically collection of similar tools under a common class named as class name + toolkit
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class AddToolInput(BaseModel):
    a: int = Field(json_schema_extra={"required" : True, "description": "First number to add"})
    b: int = Field(json_schema_extra={"required" : True, "description" : "Second number to add"})
    
class AddTool(BaseTool):
    name: str = "add"
    description : str = "add two numbers"
    
    args_schema : Type[BaseModel] = AddToolInput
    
    # Core function of the tool
    def _run(self, a: int, b: int) -> int:
        return a + b

add_tool = AddTool()
result = add_tool.invoke({"a": 30, "b": 40})
print(result)

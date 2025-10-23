# The following is the flow of using tools in langchain: define tool -> bind tool -> call tool -> execute tool -> get tool message -> pass tool message to llm

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

@tool
def divide(a: int, b: int) -> int:
    '''Given 2 numbers divide them'''
    return a / b

result = divide.invoke({"a" : 30, "b": 10})
print(result)

# Tool binding
# Only some llms have support for tool binding
llm_tool = model.bind_tools([divide])

# Tool calling : During conversation llm calls the tool if required and gives the output with name of the tool and the args it used to call the tool.
# Imp: The llm itself does not uses the tool, it just suggests the tool and pass the input variables, the actual execution of the tool is done by langchain under the hood

# Tool execution: Actual execution of code in tool by langchain to get the result

result1 = llm_tool.invoke("How is the weather today?")
result2 = llm_tool.invoke('what 200 divide by 5')
print(result1)
print(result2)



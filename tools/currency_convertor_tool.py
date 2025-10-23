# This tools converts the currency according to the real time currency rates

from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import OllamaLLM, ChatOllama
from dotenv import load_dotenv
import requests
import json
load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text_generation"
# )

# model = ChatHuggingFace(llm = llm)

llm = ChatOllama(model="qwen3:1.7b")

# Tool defining
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    '''This function fetches the current conversion factor between a base currency and target currency'''
    url = f"https://v6.exchangerate-api.com/v6/df466deb6ab6b541f2162ffb/pair/{base_currency}/{target_currency}"
    
    response = requests.get(url)
    return response.json()['conversion_rate']

result = get_conversion_factor.invoke({"base_currency": 'USD', "target_currency": "INR"})

@tool
# Annotaed[float, InjectedToolArg] -> Tells llm explicitly not to set the value of this argument
def convert_currency(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    '''This returns the equivalent value of the base currency to its equivalent target currency using the conversion_rate'''
    return base_currency_value * conversion_rate

result2 = convert_currency.invoke({"base_currency_value": 2, "conversion_rate": result})
    
    
print(result)
print(result2)

# Tool binding

model_with_tools = llm.bind_tools([get_conversion_factor, convert_currency])

# Tool calling
messages = [HumanMessage("What is the conversion factor between USD and INR, and based on it convert 30000 USD to INR")]

ai_message = model_with_tools.invoke(messages)
messages.append(ai_message)

print(ai_message.tool_calls)

for tool_call in ai_message.tool_calls:
    # execute the first tool to get value of conversion rate
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        print(tool_message1)
        conversion_rate = json.loads(tool_message1.content)
        print(conversion_rate)
        messages.append(tool_message1)
    
    # use the value of conversion rate and then find the equivalent value in target currency using second tool
    else : 
        # First inject the value of conversion rate into the function args
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert_currency.invoke(tool_call)
        messages.append(tool_message2)
    
# Final call to llm with updated values
final_result = model_with_tools.invoke(messages)
print(final_result.content)

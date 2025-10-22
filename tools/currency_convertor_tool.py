# This tools converts the currency according to the real time currency rates

from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import requests
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text_generation"
)

model = ChatHuggingFace(llm = llm)


# Tool defining
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    '''This function fetches the current conversion factor between a base currency and target currency'''
    url = f"https://v6.exchangerate-api.com/v6/df466deb6ab6b541f2162ffb/pair/{base_currency}/{target_currency}"
    
    response = requests.get(url)
    return response.json()['conversion_rate']

result = get_conversion_factor.invoke({"base_currency": 'USD', "target_currency": "INR"})

@tool
def convert_currency(base_currency_value: int, conversion_rate: float) -> float:
    '''This returns the equivalent value of the base currency to its equivalent target currency using the conversion_rate'''
    return base_currency_value * conversion_rate

result2 = convert_currency.invoke({"base_currency_value": 2, "conversion_rate": 88.0293})
    
    
print(result)
print(result2)

# Tool binding

model_with_tools = model.bind_tools([get_conversion_factor, convert_currency])

# Tool calling
messages = [HumanMessage("What is the conversion factor between USD and INR, and based on it covert 30000 USD to INR")]
print(messages)

ai_message = model_with_tools.invoke(messages)
print(ai_message)
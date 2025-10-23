from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests

llm = ChatOllama(model="qwen3:1.7b")

search_tool = DuckDuckGoSearchRun()
print(search_tool.invoke("WHat is the conversion rate of USD to IND today?"))
print(llm.invoke("How are you?"))

@tool
def get_weather_details(city: str) -> str:
    '''This method returns the current weather of the city'''
    result = requests.get(f"http://api.weatherstack.com/current?access_key=84611e68c41782f8583e239c828219a4&query={city}")
    return result
# Importing libraries for creating agents
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
# ReAct agent : Reasoning + Action (Most famous design pattern for designing agenrs)

# This line is used to import predifined prompts from langchain 

SYSTEM_PROMPT = """You are an expert web researcher, who speaks in puns.

You have access to two tools:

- search_tool: use this to get latest information from the internet

If a user asks you for a query, if dont have the latest information, use the search_tool to get the latest information and return it to the user."""

# Creating the agent
agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_details]
)

# Agent thinks what to do, agent executor executes it with the help of provided tools and returns result to the agent



response = agent.invoke({"messages": [{"role": "user", 'content': 'Whats the weather of Thailand today'}]})

print(response)
# This file contains example of a search tool in langchain
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults, ShellTool

# Search tool
search_tool = DuckDuckGoSearchResults()
search_result = search_tool.invoke("Latest stock market news")
print(search_result)

# Shell tool
# shell_tool = ShellTool()
# shell_result = shell_tool.invoke("pwd")
# print(shell_result)

from langchain_core.tools import tool
import math

# Steps to create a tool: define a function -> add type hinting across the function and message for the prompt -> use tool decorator to use it as a langchain tool -> use invoke method to activate the tool and get the result
@tool
def calculate_circle_area(radius: int) -> int:
    '''Returns the area of the circle given its radius'''
    return math.pi *radius * radius

area = calculate_circle_area.invoke({"radius": 3})
print(area)
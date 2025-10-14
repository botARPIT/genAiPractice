# This file is just for understanding the type dict format used for structured output of the llm
# This is used by the code editor for highlighting the type differences if exists, it doesnt generate any error on wrong types and is not used for validation

from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    
new_person: Person = {"name": "Bhoot", "age": 23}
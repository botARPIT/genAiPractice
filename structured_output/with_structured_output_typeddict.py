# This file contains code to direct the output of the LLM into a defined dictionary with types
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
# The type dict failed here, because the gemini's internal tool calling expects the Review to be a callable (a function) and here since its not a callable just a hint, hence inspect.signature(Review) which is used to check if a particular args or objects and since TypedDict is just a module for typeHinting system in python, it does not have a signature and hence it throws a error

import inspect
from pydantic import BaseModel, Field

load_dotenv()

# Define output schema
class Review(BaseModel):
    key_themes : Annotated[list[str], "Write the key themes discussed in the review in a list"]
    summary: Annotated[str, "Write a short summary about the product review"]
    sentiment: str = Field(description="Write sentiment of the review in one word either positve, negative or neutral")
    pros: Annotated[Optional[list[str]], "Return the pros mentioned in the review in the form of list"]
    cons: Annotated[Optional[list[str]], "Return the cons mentioned in the review in the form of list"]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_model = model.with_structured_output(Review)
print(inspect.signature(BaseModel))
result = structured_model.invoke(""" The laptop performs decently for my machine learning experiments. I can train smaller transformer models locally, but it does throttle under sustained loads. While the RTX 3050 GPU is adequate for basic deep learning, it struggles with larger models. The battery life drops quickly during heavy tasks, and the fans get loud. On the positive side, setup on Linux was easy, and the keyboard and screen are great for long coding sessions. It’s a good mid-tier option if you’re just starting out, but not ideal for advanced training. """)

print(result)
# It's a basic terminal based chatbot
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Integrating langchain messages package
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Using array to provide context to the chat model
chat_history = [
    SystemMessage(content="You are a helpful ai assistant"),
]

while True:
    user_input = input("You: ") 
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"Ai: {result.content}")
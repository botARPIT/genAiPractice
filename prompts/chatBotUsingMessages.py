''' This is used to build a chat bot that has chat history of static messages such as 
static system message, static human message and static ai message using langchains message library 
'''
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about nividia cuda cores")

]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)
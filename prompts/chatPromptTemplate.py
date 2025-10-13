''' This is used to build a chat bot that has chat history of dynamic messages such as 
dynamic system message, dynamic human message and static ai message using langchains message library 
'''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms what is {topic}')
])

prompt = chat_template.invoke({'domain': 'Telecommunication dept', 'topic': 'dvd vs vc player'})
result = model.invoke(prompt)
print(result.content)
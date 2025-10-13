''' This is basically a chat bot that uses chat placeholder to get the context from the previous chat history and then helps the consumer
with his queries
'''
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer care assistant, you job is to understand customer queries and provide resolution'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
    
])
chat_history = []
# load chat history
with open('prompts/chat_history.txt') as f:
    chat_history.extend(f.readlines())
# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'What is the current status of my order with id #234 and #1234'})

result = model.invoke(prompt)
print(result.content)
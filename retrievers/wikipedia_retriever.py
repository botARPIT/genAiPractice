from langchain_community.retrievers import WikipediaRetriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)


retriever = WikipediaRetriever(
    top_k_results = 1,
    lang = 'en'
)

query = "Why sri lanka is extremely important to India with perspective of trade and defence?"

docs = retriever.invoke(query)

for single_doc in docs:
    print('====New Content====')
    print(single_doc.page_content)
    
prompt = PromptTemplate(
    template = "Answer the following questions by pulling the data from wikipedia \n {question}",
    input_variables = ['question']
)

parser = StrOutputParser()

chain = prompt | retriever | model | parser

result = chain.invoke({'question': 'Which countries are least affected by climate change and why?'})

print(result)
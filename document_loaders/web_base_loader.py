# Web base loader is used to extract text and content from web pages, ot uses two libraries under the hood one is http library to make http request to the web page and the other is beautifulsoap to understand the website structure, it mainly works well with static web pages not with javascript heavy websites
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7w4hn-a/p/itm2ea42dec44bca?pid=COMH64PYZU4ZZR79&lid=LSTCOMH64PYZU4ZZR79AHLYXY&marketplace=FLIPKART&cmpid=content_computer_22927808323_g_8965229628_gmc_pla&tgi=sem,1,G,11214002,g,search,,770553264708,,,,c,,,,,,,&entryMethod=22927808323&&cmpid=content_22927808323_gmc_pla&gad_source=1&gad_campaignid=22927808323&gbraid=0AAAAADxRY5_fu5xnVUHfqq26DfiYm9d5a&gclid=Cj0KCQjw9czHBhCyARIsAFZlN8RAUfQG200U-hbmanQ5-cZMBk6zP6BriagaUpc2n_gthtt-4eOZWn4aArckEALw_wcB"

loader = WebBaseLoader(url)
doc = loader.load()
print(doc)

parser = StrOutputParser()
prompt = PromptTemplate(
    template = "Analyse the content of the website and generate a summary of what its about \n {content}",
    input_variables = ['content']
)

chain = prompt | model | parser
response = chain.invoke({"content" : doc[0].page_content})
print(response)



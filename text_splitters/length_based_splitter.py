# Performs text splitting based on its length

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

text = '''
    This is a normal text, just to check the working of the text splitter class 
'''

splitter = CharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_text(text)
print(result)

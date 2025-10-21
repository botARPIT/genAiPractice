# This is a kind of chatbot made for yt video specifically where the main goal is to answer the user's query related to the contents in the yt video

'''
Since its a RAG based the whole flow will be something like this:

Indexing -> Retrieval -> Augmentation -> Generation

Components of Indexing:

Document loaders -> Text splitters -> Vector store -> Retrievers

The flow of the app is:

Load video from the yt -> Load the transcription data -> Embed the data -> Store it in the vector store
-> On getting the prompt -> Embed the prompt -> Retrieve the relevant chunck from yt video -> Combine prompt from user and retrieved data
-> Passed both of them to the LLM -> Generate the response

'''

# Getting the transcript data from the yt
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Loading transciption from yt video
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=YFkeOBqfQBw",
    # add_video_info = True, -> Throws error dont know y
    transcript_format = TranscriptFormat.TEXT,
    language = ["en"]
)
# Getting the data from yt transcript api
# from youtube_transcript_api import YouTubeTranscriptApi
# ytt_api = YouTubeTranscriptApi()
# result = ytt_api.fetch("YFkeOBqfQBw")
# print(result)

transcribed_script = loader.load()
# for transcription in transcribed_script:
#     print("================================"),
#     print(transcription)

# print("\n\n".join(map(repr, transcribed_script)))
# print(type(transcribed_script))
# # print("\n\n".join(map(repr, loader.load())))
# transcribed_documents = []
# transcribed_documents.extend(transcribed_script)
# print(transcribed_documents)

# content = []

# for line in transcribed_script:
#     print(line.page_content)
#     content.append(line.page_content)

# Splitting the documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = splitter.split_documents(transcribed_script)
# Setting up an embedding model
embedding_model = HuggingFaceEmbeddings( model_name = "sentence-transformers/all-MiniLM-L6-v2")

#Setting up llm
llm = HuggingFaceEndpoint(
     repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

# Creating vector store (configuration)
vector_store = Chroma(
    embedding_function = embedding_model,
    persist_directory = "rag/yt_transcripts",
    collection_name = "YoutubeTranscripts" 
)

# Adding data to vector store

vector_store.from_documents(
    chunks,
    embedding_model  
)

print(vector_store)

print(vector_store.get(include = ['metadatas', 'embeddings', 'documents']))

query = "Which yt channel is being referred to in this video?"

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k" : 2, "lambda_mult" : 0.4} 
)

print(retriever.invoke(query))

prompt = PromptTemplate(
    template = "Answer the following query {query} from the transcription of the yt video {transcription}",
    input_variables = ['query', 'transcription']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"query" : query, "transcription": retriever.invoke(query)})


print("From LLM", result)
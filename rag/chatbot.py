from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

video_id = "nKSk_TiR8YA"

try:
    loader = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=YFkeOBqfQBw",
        transcript_format = TranscriptFormat.TEXT,
        language = ["en"]
    )
    
    transcription = loader.load()

except:
    print("No captions available for the video")
    
    
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 750,
    chunk_overlap = 50
)

chunks = splitter.split_documents(transcription)


embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)

print(vector_store.index_to_docstore_id)

retriever = vector_store.as_retriever( search_type = "similarity", search_kwargs = {"k" : 3})

print(retriever.invoke("What is this video about?"))

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = '''
    You are a helpful assistant. Answer ONLY from the provided context.
    If the context is insufficient, just say you don't know.
    {context}
    Question : {question}
    ''',
    input_variables=['context', 'question']
)

parser = StrOutputParser()
question  = "Which yt channel is being referred to in this video?"

retrieved_docs = retriever.invoke(question)

def generate_context(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'question': RunnablePassthrough(),
    'context' : retriever | RunnableLambda(generate_context)
})

simple_chain = prompt | model | parser
chain = parallel_chain | simple_chain
response = chain.invoke(question)
print(response)
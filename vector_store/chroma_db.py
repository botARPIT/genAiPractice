from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

embedding = HuggingFaceEmbeddings( model_name = "sentence-transformers/all-MiniLM-L6-v2" )

documents = [
    Document(
        page_content="Artificial intelligence is transforming the software industry, enabling faster automation and smarter analytics.",
        metadata={'topic': 'AI & Automation', 'source': 'Tech Journal', 'year': 2024}
    ),
    Document(
        page_content="Cloud-native applications rely heavily on containerization and orchestration tools such as Docker and Kubernetes.",
        metadata={'topic': 'Cloud Computing', 'source': 'DevOps Digest', 'year': 2025}
    ),
    Document(
        page_content="Quantum computing promises exponential speed improvements for specific problems like cryptography and optimization.",
        metadata={'topic': 'Quantum Tech', 'source': 'Science Daily', 'year': 2025}
    ),
    Document(
        page_content="Edge computing reduces latency by processing data closer to users, improving real-time performance in IoT applications.",
        metadata={'topic': 'Edge & IoT', 'source': 'IoT Weekly', 'year': 2024}
    ),
    Document(
        page_content="Natural language models like GPT are becoming integral to customer support, creative writing, and coding assistance.",
        metadata={'topic': 'Generative AI', 'source': 'AI Today', 'year': 2025}
    ),
    Document(
        page_content="Cybersecurity practices now focus on zero-trust frameworks, continuous authentication, and behavior analytics.",
        metadata={'topic': 'Security', 'source': 'Cyber Insights', 'year': 2025}
    ),
    Document(
        page_content="Renewable energy technologies, supported by AI-driven forecasting, are improving grid stability and sustainability.",
        metadata={'topic': 'Green Tech', 'source': 'Energy Future', 'year': 2024}
    ),
    Document(
        page_content="Augmented reality is reshaping training, design, and retail experiences by overlaying digital information on real environments.",
        metadata={'topic': 'AR/VR', 'source': 'Tech Visuals', 'year': 2023}
    )
]

vector_store = Chroma(
    embedding_function = embedding,
    persist_directory = 'vector_store/chroma_db',
    collection_name = 'testing'
)

# Adding documents to the vector store
vector_store.add_documents(documents)

print(vector_store.get(include = ['embeddings', 'documents', 'metadatas']))

search_query = vector_store.similarity_search(
    query = 'What are the benefits of edge computing?',
    k = 1
)

print("Search query", search_query)
# This is an advanced retriever which improves the retrieval quality by compressing the documents after retrieval

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings( model_name = "sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)


sample_data = [
    Document(page_content=(
        "The rapid adoption of blockchain in finance has improved transaction transparency, "
        "but similar principles are now being tested in supply chain traceability for agriculture. "
        "At the same time, advances in machine learning are enhancing climate forecasting models, "
        "helping governments allocate resources more efficiently. "
        "Meanwhile, universities are using cloud-based simulations to teach economics students about decentralized finance."
    )),
    
    Document(page_content=(
        "AI-powered investment platforms are reshaping how portfolios are managed, yet the same neural network architectures "
        "are being used by hospitals to predict patient recovery times. "
        "While fintech startups continue to expand digital payment ecosystems, the renewable energy sector is adopting predictive analytics "
        "to optimize power grid stability. "
        "Even education platforms are borrowing gamification models from trading apps to keep learners engaged."
    )),
    
    Document(page_content=(
        "Cybersecurity threats in global banking have prompted new encryption standards that also benefit e-commerce and smart city systems. "
        "The insurance industry, meanwhile, is using computer vision to assess property damage, a technique derived from autonomous vehicles. "
        "In parallel, researchers are applying similar models to analyze satellite imagery for environmental monitoring. "
        "All these innovations depend on robust cloud infrastructure originally built for large-scale social networks."
    )),
    
    Document(page_content=(
        "Quantum computing, once purely academic, is now being explored by hedge funds for rapid market optimization, "
        "while pharmaceutical companies are using it to accelerate drug discovery. "
        "At the same time, urban planners leverage IoT data streams to manage traffic flows and reduce emissions. "
        "Meanwhile, digital identity systems powered by blockchain are finding applications from banking KYC compliance to university credential verification."
    )),
    
    Document(page_content=(
        "The intersection of AI ethics and financial automation has raised concerns about algorithmic bias in loan approvals. "
        "In healthcare, similar ethical debates emerge around diagnostic recommendations from machine learning models. "
        "Educational institutions are now teaching data literacy as a fundamental skill, bridging topics from coding to behavioral economics. "
        "Even art markets are joining the trend, using AI to authenticate digital artwork and manage NFT transactions."
    )),
    
    Document(page_content=(
        "The growth of digital currencies has influenced how governments think about taxation and fiscal policy, "
        "while logistics companies apply blockchain to ensure transparent tracking of goods. "
        "Meanwhile, predictive maintenance powered by IoT sensors reduces operational downtime in manufacturing plants. "
        "Energy companies are adopting similar systems to monitor offshore wind farms and optimize turbine performance."
    )),
    
    Document(page_content=(
        "Augmented reality tools designed for gaming are now being tested by banks for immersive data visualization dashboards. "
        "Healthcare professionals use AR-assisted training to simulate surgical procedures, "
        "while architects employ the same technology to preview urban development projects. "
        "In another domain, AI-based sentiment analysis tracks how consumers react to such innovations across social media."
    )),
    
    Document(page_content=(
        "Digital twins are revolutionizing urban planning by simulating infrastructure behavior under stress, "
        "while insurance companies use similar models to predict claim risks. "
        "Financial institutions are integrating these simulations into risk management frameworks, "
        "and the entertainment industry is experimenting with them to model audience engagement trends. "
        "The underlying data pipelines often rely on big data frameworks originally designed for ad tech analytics."
    )),
    
    Document(page_content=(
        "The push toward carbon-neutral operations is reshaping how financial institutions assess green investments, "
        "while transportation firms electrify fleets to reduce emissions. "
        "AI models are being trained to monitor deforestation rates and inform sustainability funds. "
        "Meanwhile, blockchain startups are creating carbon credit exchanges, "
        "and universities are collaborating on open data projects that unify environmental, financial, and social metrics."
    )),
    
    Document(page_content=(
        "High-frequency trading systems rely on microsecond data transmission, similar to latency-sensitive applications in remote surgery. "
        "The same edge computing architectures serve both industries — ensuring fast decisions in markets and medicine alike. "
        "Education startups are adapting these architectures for adaptive testing systems, "
        "while logistics companies use them to optimize fleet coordination in real time."
    ))
]

vector_store = FAISS.from_documents(
    documents=sample_data,
    embedding = embedding_model,
)

base_retriever = vector_store.as_retriever(search_kwargs = {"k" : 3})

# Compressor using llm
compressor_llm = LLMChainExtractor.from_llm(model)

# Contextual compression retriever

compression_retriever = ContextualCompressionRetriever(
base_retriever= base_retriever,
base_compressor= compressor_llm
)

query = "Machine laerning advancements in climate forecasting"

result = compression_retriever.invoke(query)

print(result)
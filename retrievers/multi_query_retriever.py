from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.retrievers import MultiQueryRetriever

embedding_model = HuggingFaceEmbeddings( model_name = "sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

documents = [
    Document(page_content="The rise of fintech startups has revolutionized the banking sector, enabling faster digital transactions and seamless online account management."),
    Document(page_content="Blockchain technology is transforming financial systems by providing transparent, tamper-proof records for cross-border transactions."),
    Document(page_content="AI-driven fraud detection systems now analyze millions of transactions per second to identify suspicious patterns in real time."),
    Document(page_content="Neobanks are gaining popularity among millennials due to their user-friendly mobile interfaces and zero-fee banking models."),
    Document(page_content="Robo-advisors are reshaping investment management by offering algorithm-driven portfolio balancing with minimal human intervention."),
    Document(page_content="Digital payment platforms like UPI and PayPal have drastically reduced the dependency on physical cash across developing economies."),
    Document(page_content="Central banks worldwide are experimenting with Central Bank Digital Currencies (CBDCs) to modernize national payment infrastructures."),
    Document(page_content="High-frequency trading firms use advanced algorithms to execute trades within microseconds, leveraging small price fluctuations for profit."),
    Document(page_content="Open banking APIs are enabling financial institutions to securely share user data with third-party developers for innovative service creation."),
    Document(page_content="Insurance companies are increasingly using machine learning to optimize risk models and offer personalized premium plans."),
    Document(page_content="Cryptocurrency adoption continues to grow as investors view digital assets as both speculative opportunities and inflation hedges."),
    Document(page_content="Micro-investment platforms are allowing users to invest spare change from everyday purchases into diversified portfolios."),
    Document(page_content="RegTech solutions are helping financial institutions automate compliance reporting and reduce the risk of regulatory violations."),
    Document(page_content="Decentralized finance (DeFi) is disrupting traditional banking by enabling peer-to-peer lending and borrowing without intermediaries."),
    Document(page_content="AI chatbots in banking now handle routine customer queries, improving efficiency and reducing operational costs."),
    Document(page_content="Stock trading apps have democratized investing, giving retail investors access to tools once reserved for professionals."),
    Document(page_content="Credit scoring models are increasingly using alternative data like mobile payment history to assess borrower reliability."),
    Document(page_content="Sustainable finance initiatives are encouraging banks to prioritize green projects and reduce their carbon footprints."),
    Document(page_content="Wealth management platforms integrate AI insights to deliver hyper-personalized financial planning experiences."),
    Document(page_content="Predictive analytics helps hedge funds forecast market movements and optimize trading strategies with greater accuracy.")
]

documents1 = [
    Document(page_content="AI-driven fraud detection systems are transforming modern banking by analyzing transactional data in real time to identify unusual spending patterns."),
    Document(page_content="Artificial intelligence is helping banks prevent fraud by detecting abnormal transaction behaviors before losses occur."),
    
    Document(page_content="Blockchain technology ensures transparent and immutable financial records, significantly reducing fraud in cross-border payments."),
    Document(page_content="The use of blockchain in banking is increasing as institutions aim to improve transaction security and transparency."),
    
    Document(page_content="Neobanks are redefining retail banking by providing mobile-first experiences, lower fees, and instant digital onboarding."),
    Document(page_content="Digital-only banks, or neobanks, appeal to younger consumers with fast, app-based banking and real-time balance updates."),
    
    Document(page_content="High-frequency trading algorithms execute thousands of trades in milliseconds to capitalize on minute market fluctuations."),
    Document(page_content="Quantitative trading firms leverage algorithmic systems to gain microsecond-level advantages in volatile financial markets."),
    
    Document(page_content="Central Bank Digital Currencies are being explored as a way to modernize monetary systems and reduce reliance on physical cash."),
    Document(page_content="CBDCs could bridge the gap between traditional fiat systems and digital payments while improving transaction traceability."),
    
    Document(page_content="Robo-advisors use AI algorithms to automate portfolio management, offering investors low-cost, data-driven financial advice."),
    Document(page_content="Automated investment platforms are becoming popular for passive investors seeking AI-based asset allocation strategies."),
    
    Document(page_content="Open banking APIs enable fintech startups to securely access user data and create personalized financial tools."),
    Document(page_content="Through open banking, customers can connect multiple accounts to third-party apps for better financial insights."),
    
    Document(page_content="RegTech solutions automate compliance monitoring, helping institutions stay aligned with constantly changing regulations."),
    Document(page_content="Financial firms use regulatory technology to simplify auditing and ensure consistent adherence to global compliance laws."),
    
    Document(page_content="Sustainable finance is on the rise as investors prioritize ESG (Environmental, Social, and Governance) factors in portfolio decisions."),
    Document(page_content="Green investments are gaining momentum as more funds integrate sustainability metrics into their risk assessment models."),
    
    Document(page_content="Predictive analytics allows hedge funds to anticipate market movements and adjust their strategies with improved accuracy."),
    Document(page_content="Machine learning models are enhancing trading predictions by learning from years of historical price and sentiment data."),
    
    Document(page_content="Micro-investing apps allow users to invest small amounts regularly, promoting financial inclusion and long-term wealth building."),
    Document(page_content="Fractional investing is empowering users to own small portions of high-value stocks, making wealth creation more accessible."),
]

vector_store = FAISS.from_documents(
    documents= documents1,
    embedding = embedding_model,
)

# Similarity retriever
similarity_retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {"k" : 5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_kwargs = { "k" : 5}),
    llm = model
)

query = "Trends in finance industry"

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

print(similarity_results)
print("=================================")
print(multiquery_results)

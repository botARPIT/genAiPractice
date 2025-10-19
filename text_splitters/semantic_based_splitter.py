from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding =  HuggingFaceEmbeddings( model_name = "sentence-transformers/all-MiniLM-L6-v2")

sample = '''
The sun crept over the skyline, brushing the buildings with a faint golden hue. Commuters filled the streets, their footsteps echoing like a rhythmic prelude to the city’s daily pulse.
Deep beneath those streets, servers hummed quietly in a temperature-controlled chamber. They processed millions of API requests per second, keeping countless applications alive and synchronized.
A farmer, miles away, examined his crops through a drone’s live camera feed. Precision agriculture allowed him to predict soil health and irrigation needs more accurately than ever before.
Meanwhile, in a classroom filled with digital tablets, students learned history through interactive simulations. Ancient civilizations came to life in augmented reality, making lessons immersive and unforgettable.
On the other side of the globe, a small development team deployed a new machine learning model. They watched the logs anxiously, hoping the new version wouldn’t introduce latency or unpredictable bias.
In a nearby café, two writers debated the ethics of AI-generated art. One believed creativity was uniquely human; the other saw it as a collaboration between human intent and machine potential.
A surgeon, preparing for an operation, reviewed a 3D holographic scan of the patient’s heart. Every vessel and artery was visible in real time, ensuring precision and safety.
Outside the hospital, an electric bus glided down the road in near silence. The air felt cleaner than it had in years, thanks to improved emission standards and renewable energy grids.
In a quiet home office, a software engineer sipped coffee while debugging a caching issue. Each microservice call had to perform optimally under load, or the entire pipeline would stall.
Across town, a novelist struggled to describe the emotion of loss in her latest manuscript. Words felt inadequate, but persistence kept her fingers moving across the keyboard.
A robotic arm in a warehouse lifted boxes with mechanical grace. Its sensors detected depth, balance, and fragility — preventing damage and ensuring efficiency.
Elsewhere, a group of children built a small wind turbine using scrap materials. Their laughter mixed with the hum of spinning blades, a simple experiment turning into a powerful lesson.
At a startup incubator, entrepreneurs pitched their prototypes to investors. Some ideas were practical, others wildly ambitious, but all carried the same spark of human ingenuity.
In the Arctic, scientists measured the retreat of glaciers with a mix of awe and concern. Their data painted a sobering picture of how rapidly the planet’s climate was transforming.
A poet stood by a river at dusk, notebook in hand. The reflection of stars on the water reminded her that beauty often hides in quiet observation.
Moments later, a cybersecurity analyst received an alert about an unusual login attempt. Within seconds, he isolated the threat and secured the system before damage could occur.
The night settled again over the city, lights flickering like neurons in a vast digital brain. Somewhere between the noise and the silence, progress continued — steady, invisible, and deeply human.
'''
text_splitter = SemanticChunker(
    embedding, 
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs[1])
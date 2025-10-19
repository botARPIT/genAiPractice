# This is the second type of spiltter that works based on the structure of the textual data, the basic concept of this splitter is to split the document first on the basis of paragraph, then on basis of sentence, then by words and lastly by characters in order to split it based on the chunk size mentioned

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

text = '''
The concept of technology continues to evolve as new ideas emerge, reshaping the way people think and interact with the world around them. Innovation drives creativity, pushing society to explore new ways of solving old problems. In nature, the balance between growth and decay mirrors the constant rhythm of progress and renewal found in human life. Education plays a vital role in this transformation, empowering individuals with knowledge and critical thinking skills that open doors to endless possibilities. Science, meanwhile, provides the structure and evidence that turn imagination into tangible discoveries.

Travel exposes people to different perspectives, helping them appreciate the vast diversity that defines human civilization. Literature captures these experiences, transforming emotions and thoughts into written words that outlast generations. History reminds us of where we have been, guiding our path forward through lessons both painful and inspiring. The future, though uncertain, always invites curiosity and courage — a chance to build something better than what came before.

Technology today shapes nearly every aspect of modern living, from communication to medicine to transportation. The boundaries between physical and digital spaces continue to blur as artificial intelligence, robotics, and automation redefine efficiency. Education adapts alongside these changes, integrating new methods of learning and collaboration. Online classrooms, interactive simulations, and adaptive testing reflect how innovation reshapes traditional institutions. The idea of lifelong learning has become more relevant than ever, as skills evolve faster than systems can formalize them.

Nature, despite humanity’s rapid advancement, remains both a source of inspiration and a reminder of humility. The resilience of ecosystems shows how balance and adaptation are key to survival. Science takes these principles and applies them to human problems — from renewable energy to climate change mitigation. Innovation thrives when technology and nature coexist rather than compete.

Society stands at the intersection of all these forces. Literature documents the emotional pulse of change, giving voice to struggles and triumphs. The history of progress is rarely linear; it is shaped by trial, error, and perseverance. Each generation inherits the outcomes of the last — sometimes as blessings, sometimes as burdens. But progress depends not only on invention, but also on reflection.

In the future, education will likely merge with technology even more deeply, creating immersive learning environments that blend physical and digital experiences. Science will continue to uncover the mysteries of the universe, while innovation turns those discoveries into tools that improve daily life. Nature will persist as both muse and mirror, reminding humanity that all advancement carries responsibility. Society will evolve, guided by stories, knowledge, and shared imagination.

Ultimately, the relationship between technology, nature, science, and education forms the foundation of human progress. Each complements the other, weaving a narrative that connects curiosity to creation. The future remains unwritten, but within that uncertainty lies endless potential — a promise that with imagination, empathy, and persistence, humanity will continue to grow.
'''


splitter = RecursiveCharacterTextSplitter( 
    chunk_size = 300,
    chunk_overlap = 0
    )

docs = splitter.split_text(text)

print(len(docs))
print(docs[1])
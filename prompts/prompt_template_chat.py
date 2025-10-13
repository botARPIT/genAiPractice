# This app uses the prompt template class for single message of the langchain, to provide context to the model to get better results
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
st.header("Document summarizer")

paper_input = st.selectbox("Select the document", ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers", 
    "GPT3: New Language Models", "The Era of diffusion models"])

style_input = st.selectbox("Select explanation style", ["Beginner", "Technical", "Code oriented", "Expert"])

length_input = st.selectbox("Select explanation length", ["Short (1-2 Paragraph)", "Medium (3-4 Paragraph)", "Long (detailed explanation)"])

# # Template
# template = PromptTemplate(
#     template = """
#     Please summarize the research paper titled "{paper_input}" with the following specifications:
#     Explanation style: {style_input}
#     Explanation length : {length_input}
#     1. Mathematical details: 
#         - Include relevant mathematical equations if present in the paper.
#         - Explain the mathematical concepts using simple, intiutive code snippets where applicable.
#     2. Analogies:
#         - Use relevant analogies to simplify complex ideas
#     If certain informtation is unavailabe in the paper, reply with "Insufficient information available" instead of guessing. 
#     Ensure the summary is clear, accurate and aligned with the provided style and length
#     """,
    
#     input_variables=['paper_input', 'style_input', 'length_input'],
#     validate_template=True
# )

template = load_prompt('template.json')
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
# This generates the prompt template in json
from langchain_core.prompts import PromptTemplate

# Template
template = PromptTemplate(
    template = """
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation style: {style_input}
    Explanation length : {length_input}
    1. Mathematical details: 
        - Include relevant mathematical equations if present in the paper.
        - Explain the mathematical concepts using simple, intiutive code snippets where applicable.
    2. Analogies:
        - Use relevant analogies to simplify complex ideas
    If certain informtation is unavailabe in the paper, reply with "Insufficient information available" instead of guessing. 
    Ensure the summary is clear, accurate and aligned with the provided style and length
    """,
    
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template=True
)

template.save('template.json')
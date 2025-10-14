from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
# Json schema (blueprint) for the customer review
review_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
       
        "items": {
            "type: string"
        },
        "description": "Write down all the key themes discussed in the review in a list"
    }, 
       "summary": {
           "type": "string",
           "description": "Brief summary on the review"
       },
       "sentiment":{
           "type": "string",
           "enum": ["positive", "negative", "neutral"],
           "description": "Return sentiment of the review either positive, negative or neutral"
       }, 
       "pros":{
           "type": ["array", "null"],
           "items": {
               "type": "string",
               "description": "Write all the pros mentioned about the product in a list"
           }
       },
       "cons":{
           "type": ["array", "null"],
           "items": {
               "type": "string",
               "description": "Write all the cons mentioned about the product in a list"
           }
       },
       "name": {
           "type": ["string", "null"],
           "description": "Write the name of the reviewer"
       }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(review_schema)

output = structured_model.invoke(""" The laptop performs decently for my machine learning experiments. I can train smaller transformer models locally, but it does throttle under sustained loads. While the RTX 3050 GPU is adequate for basic deep learning, it struggles with larger models. The battery life drops quickly during heavy tasks, and the fans get loud. On the positive side, setup on Linux was easy, and the keyboard and screen are great for long coding sessions. It’s a good mid-tier option if you’re just starting out, but not ideal for advanced training. """)

print(output)

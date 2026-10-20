from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables from .env
load_dotenv()

# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini")

# System Prompt: Persona, formatting, and constraints
system_prompt = (
    "You are an expert nutritionist and meal planning assistant. "
    "Provide exactly 5 healthy breakfast options. "
    "Output strictly as a numbered list from 1 to 5, one item per line. "
    "Do not include any introductory remarks, explanations, or concluding text."
)

# User Prompt: Task details and specific item constraint
user_prompt = (
    "Provide the breakfast list. "
    "Item 1 must be 'Vegetable oats upma with peanuts'."
)

# Combine into messages list
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt),
]

# Invoke the model and print the result
reply = llm.invoke(messages)
print(reply.content)
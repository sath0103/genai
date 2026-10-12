from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
prompts = [" what is python", " what is machine learning", " what is data science"]
for p in prompts:
 print(llm.invoke(p).content)

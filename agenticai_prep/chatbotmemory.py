import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

os.environ["GROQ_API_KEY"]= os.getenv("GROQ_API_KEY")

model=ChatGroq(model="openai/gpt-oss-120b")

# print(model.invoke("hello"))



import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=api_key)
chat_model = ChatOpenAI()
text = "What would be a good company name for a company that makes colorful socks?"

print(f"LLM: {llm.predict(text)}")
print(f"Chat_Model: {chat_model.predict(text)}")

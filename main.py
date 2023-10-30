import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=api_key, model_name="text-davinci-003")
text = "Explain Large Language Models in one sentence"

print(f"LLM: {llm(text)}")

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(
        content="Write a Python script that trains a neural network on simulated data "
    ),
]

response = chat(messages)
print(f"ChatOpenAI: {response.content}", end="\n")

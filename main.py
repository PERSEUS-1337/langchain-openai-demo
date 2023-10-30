import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Basic OpenAI
llm = OpenAI(openai_api_key=api_key, model_name="text-davinci-003")
# text = "Explain Large Language Models in one sentence"
# print(f"LLM: {llm(text)}")

# Basic Prompting
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
# messages = [
#     SystemMessage(content="You are an expert data scientist"),
#     HumanMessage(
#         content="Write a Python script that trains a neural network on simulated data "
#     ),
# ]
# response = chat(messages)
# print(f"ChatOpenAI: {response.content}", end="\n")

# Prompt Templates
template = """You are an expert data scientist with an expertise in building deep learning models. Explain the 
concept of {concept} in a couple of lines."""
prompt = PromptTemplate(input_variables=["concept"], template=template)
print(prompt)

print(f"LLM: {llm(prompt.format(concept='regularization'))}")

# Chains
# chain = LLMChain(llm=llm, prompt=prompt)

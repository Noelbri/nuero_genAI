from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
import requests
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#define the mock APIs for demand and competitor pricing
def get_demand_data(product_id:str)->dict:
    """Mock demand API to get demand data for a product"""
    #In real use, replace with an actual API call
    return {"product_id": product_id, "demand_level":"high"}
def get_competitor_pricing(product_id:str)->dict:
    """Mock competitor pricing API"""
    return {"product_id": product_id, "competitor_price": 95.0}
#list of tools for the agent to call
tools = [get_demand_data, get_competitor_pricing]
#Define the agent using the ReAct pattern
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
graph = create_react_agent(model, tools=tools)
#Define the initial state of the agent
initial_messages = [
    ("system", "You are an AI agent that dynamically adjusts product prices based on market demand and competitors prices."),
    ("user", "What should be the price for product ID '12345'?")
]
#Stream agent responses and decisions
inputs = {"messages": initial_messages}
#Simulate the agent reasoning, acting (calling tools), and observing
for state in graph.stream(inputs, stream_mode="values"):
    message = state["messages"][-1]
    if isinstance(message, tuple):
        print(message.content)
    else:
        message.pretty_print()


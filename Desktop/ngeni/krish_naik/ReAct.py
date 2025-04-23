from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#define tools
@tool
def product_info(product_name:str)->str:
    """Fetch product information"""
    product_catalog = {
        "iPhone": "The latest iPhone features an A15 chip and improved camera.",
        "MacBook": "The new MacBook has an M2 chip and a 14-inch Retina display.",
    }
    return product_catalog.get(product_name, "Sorry, product not found.")
@tool
def check_stock(product_name:str)->str:
    """Check the stock status of a given product"""
    stock_data ={
        "iPhone": "In stock",
        "MacBook": "Out of stock",
    }
    return stock_data.get(product_name, "Stock information unavailable")
#Initialize the memory saver for single-thread memory
checkpointer = MemorySaver()
#Initialize llm
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
#create the ReAct agent with the memory saver
tools=[product_info, check_stock]
graph = create_react_agent(model=llm, tools=tools, checkpointer=checkpointer)
#set up the thread configuration to simulate single-threaded memory
config = {"configurable":{"thread_id":"thread_1"}}
#user input:initial inquiry
inputs = {"messages": [("user", "Hi, I'm Noel. Tell me about the new iPhone.")]}
messages = graph.invoke(inputs, config=config) 
for message in messages["messages"]:
    print(message.content)
#User input:repeated inquiry (memory recall)
inputs2 = {"messages":[("user", "Is the new iPhone in stock.")]}
messages2 = graph.invoke(inputs2, config=config)
for message in messages2["messages"]:
    print(message.content)
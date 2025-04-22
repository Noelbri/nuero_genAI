from dotenv import load_dotenv
import os 
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessageChunk
import operator
from typing import Annotated
from typing_extensions import TypedDict
import asyncio
from langgraph.graph.message import add_messages
#Load envirnment variables from .env file
load_dotenv
#Get the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")
#define the state schema
class State(TypedDict):
    messages:Annotated[list, add_messages]
#Initialize the LLM
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
#define a node to handle LLM queries 
async def call_llm(state:State):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages":[response]}
#define the graph
workflow = StateGraph(State)
workflow.add_node("call_llm", call_llm)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)
app = workflow.compile()
#simulate interaction and stream token
async def simulate_interaction():
    input_message = {"message":[("human", "tell me a very long joke")]}
    first = True
    # Stream LLM tokens
    async for msg, metadate in app.astream(input_message, stream_mode="messages"):
        if msg.content and not isinstance 
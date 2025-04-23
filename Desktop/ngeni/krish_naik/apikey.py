from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
import asyncio

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Define the state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the LLM
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, streaming=True)

# Define a node to handle LLM queries 
async def call_llm(state: State):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(State)
workflow.add_node("call_llm", call_llm)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)
app = workflow.compile()

# Simulate interaction and stream tokens
async def simulate_interaction():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending the conversation.")
            break

        input_message = {
            "messages": [HumanMessage(content=user_input)]
        }

        first = True
        gathered = None

        async for msg, metadata in app.astream(input_message, stream_mode="messages"):
            if isinstance(msg, AIMessageChunk) and msg.content:
                print(msg.content, end="", flush=True)
                if first:
                    gathered = msg
                    first = False
                else:
                    gathered = gathered + msg

            if msg.tool_call_chunks:
                print(gathered.tool_calls)
        print()  # new line after each response

# Run the interaction
asyncio.run(simulate_interaction())

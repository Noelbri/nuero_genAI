from dotenv import load_dotenv
import os 
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from  langgraph.prebuilt import ToolNode, tools_condition
load_dotenv
api_key = os.getenv("GROQ_API_KEY")
# Initialize the LLM and define a basic multiplication tool
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
#define a multiplication tool
def multiply(a:int, b:int):
    """
    Multiplies two numbers.
    """
    return a*b
#bind the LLM with tools bound
llm_with_tools = llm.bind_tools([multiply])
#Node that calls the LLM with tools bound
def tool_calling_llm(state: MessagesState):
    """
    Node that calls the LLM with tools bound.
    """
    messages = state.get("messages", [])
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}  # Append to keep the conversation history
#build the workflow
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
#define edges to connect the nodes
builder.add_edge(START, "tool_calling_llm")
#add conditional edge based on tool usage
builder.add_conditional_edges("tool_calling_llm",tools_condition)
builder.add_edge("tools", END)
graph = builder.compile()
def simulate():
    user_input = {
        "messages": [HumanMessage(content="can you multiply 32 by 40")]
    }
    result = graph.invoke(user_input)
    return result["messages"][-1].pretty_print()

print(simulate())
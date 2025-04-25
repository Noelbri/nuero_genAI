from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.prebuilt import ToolNode, tools_condition
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
## Arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools =[wiki_tool, arxiv_tool]
#load the LLM
llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")
llm = llm.bind_tools(tools=tools)
class State(TypedDict):
    messages:Annotated[list, add_messages]
    
graph_builder = StateGraph(State)
def chatbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot", 
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)
graph =graph_builder.compile()
from IPython.display import Image, display
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass
while True:
    user_input = input("User:")
    if user_input.lower() in ["quit", "q"]:
        print("Goodbye")
        break
    for event in graph.stream({'messages':("user",user_input)}):
        print(event.values())
        for value in event.values():
            #print(value['messages'])
            print("Assistant:",value["messages"].content)
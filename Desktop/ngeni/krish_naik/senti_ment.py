from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableConfig
from textblob import TextBlob
import json
from dotenv import load_dotenv
import os 
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#define the state for the agent
class AgentState(TypedDict):
    """The state of the agent"""
    #'add_messages' is a reducer that manages the message sequence
    messages: Annotated[Sequence[BaseMessage], add_messages]
#define tool for sentiment analysis using TextBlob
@tool
def analyze_sentiment(feedback:str)->str:
    """Analyze customer feedback sentiment with custom logic."""
    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0.5:
        return "positive"
    elif analysis.sentiment.polarity == 0.5:
        return "neutral"
    else:
        return "negative"
    
@tool
def respond_based_on_sentiment(sentiment:str)->str:
    """Respond to the customer based on the analyzed sentiment."""
    if sentiment == "positive":
        return "Thank you for your positive feedback."
    elif sentiment == "neutral":
        return "We appreciate your feedback."
    else:
        return "We are sorry to hear that you are not satisfied. How can we help?"
tools = [analyze_sentiment, respond_based_on_sentiment]
#Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
llm = llm.bind_tools(tools)
#create a dictionary of tools by their names for easy lookup
tools_by_name = {tool.name:tool for tool in tools}
#tool node to handle sentiment analysis and response generation
def tool_node(state:AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messsages": outputs}
#LLM reasoning node to process the feedback and call tools if needed
def call_model(state:AgentState, config:RunnableConfig):
    system_prompt = SystemMessage(
        content="You are a helpful assistant tasked with responding to customer feedback."
    )
    response = llm.invoke([system_prompt] + state["messages"], config)
    return {"messages":[response]}
def should_continue(state:AgentState):
    last_message = state["messages"][-1]
    #If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
#we build the state graph to orchestrate the reasoning, tool calling and observing phases. 
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    },
)
workflow.add_edge("tools", "agent")
#compile the graph
graph = workflow.compile()
#initialize the agent's state with the user's feedback
initial_state = {"messages": [("user", "hello what is an AI agent.")]}
#Helper function to print the conversation
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
#run the agent
print_stream(graph.stream(initial_state, stream_mode="values"))

from dotenv import load_dotenv
import os 
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
#Load envirnment variables from .env file
load_dotenv
#Get the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")
#Initialize the LLM
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, api_key=api_key)
#Node function to handle user query and to call the llm
def call_llm(state:MessagesState):
    messages = state["messages"]
    response = model.invoke(messages[-1].content)
    return {"messages":[response]}
#define the graph
workflow = StateGraph(MessagesState)
#add the node to call the llm
workflow.add_node("call_llm", call_llm)
#define the edges
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)
#initialize the checkpointer for short-term memory
checkpointer = MemorySaver()
#compile the workflow
app = workflow.compile(checkpointer=checkpointer)
#Function to continuously take user input
def interact_with_agent():
    while True:
        #simulate a new session by allowing the user to input a thread ID
        thread_id = input("Enter thread ID (or'new' for a new session):")
        if thread_id.lower() in ["exit", "quit"]:
            print("Ending the conversation")
            break
        if thread_id.lower() == "new":
            thread_id= f"session_{os.urandom(4).hex()}"
        while True:
            user_input = input("You:")
            if user_input.lower() in ["exit", "quit"]:
                print("Ending the conversation.")
                break
            input_message = {
                "messages": [HumanMessage(content=user_input)]
            }
            #Invoke the graph with short-term memory enabled 
            config = {"configurable":{"thread_id":thread_id}}
            for chunk in app.stream(input_message, config=config, stream_mode="values"):
                chunk["messages"][-1].pretty_print()
#start the conversation
interact_with_agent()
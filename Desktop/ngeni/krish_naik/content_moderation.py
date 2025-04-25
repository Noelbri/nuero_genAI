import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
#Initialize llm
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=50
)
#define the structure
class State(TypedDict):
    input:str
    draft_content:str
#define node functions
def create_draft(state:State):
    print("---Generating Draft with llama---")
    #prepare the prompt for generating the blog content
    prompt = f"write a blog post on the topic: {state['input']}"
    #Call the langchain llama instance to generate the draft
    response = llm.invoke([{"role":"user", "content":prompt}])
    #extract the generated content
    state["draft_content"] = response.content
    print(f"Generated Draft:\n {state['draft_content']}")
    return state
def review_draft(state:State):
    print("--Reviewing Draft---")
    print(f"Draft for review:\n{state['draft_content']}")
    return state
def publish_content(state:State):
    print("---Publishind content---")
    print(f"Publishe Content:\n{state['draft_content']}")
    return state
#build graph
builder = StateGraph(State)
builder.add_node("create_draft", create_draft)
builder.add_node("review_draft", review_draft)
builder.add_node("publish_content", publish_content)
#define flow
builder.add_edge(START, "create_draft")
builder.add_edge("create_draft", "review_draft")
builder.add_edge("review_draft", "publish_content")
builder.add_edge("publish_content", END)
#set up memory and breakpoints
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["publish_content"])
#run the graph
config = {"configurable":{"thread_id":"thread-1"}}
initial_input = {"input":"The importance of AI in modern content creation"}
#run the first part until the review step
thread = {"configurable":{"thread_id":"1"}}
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)
#pausing for human review
user_approval = input("Do you approve the draft for publishing?(yes/no/modification):")
if user_approval.lower() == "yes":
    #proceed to publish content
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
elif user_approval.lower() == "modification":
    #Allow modification of the draft
    updated_draft = input("Please modify the draft content:\n")
    memory.update({"draft_content":updated_draft})  #update memory with new content
    print("Draft updated by the editor.")
    #continue to publishing with the modified draft
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
else:
    print("Execution halted by user.")
    

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
import finnhub

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
finnhub_api = os.getenv("FINNHUB")

# Initialize Finnhub API client
finnhub_client = finnhub.Client(api_key=finnhub_api)

# Define the tool: querying stock prices using the Finnhub API 
@tool
def get_stock_price(symbol: str):
    """Retrieve the latest stock price for the given symbol."""
    quote = finnhub_client.quote(symbol)
    return f"The current price for {symbol} is ${quote['c']}."

# Register the tool in the tool node
tools = [get_stock_price]
tool_node = ToolNode(tools)

# Set up the AI model
model = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")
model = model.bind_tools(tools)

# Define the function that simulates reasoning and invokes the model
def agent_reasoning(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Define conditional logic to determine whether to continue or stop
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, finish the process
    if not last_message.tool_calls:
        return "end"
    return "continue"  # Otherwise, continue to the next step

# Build the agent workflow using LangGraph
workflow = StateGraph(MessagesState)

# Add nodes: agent reasoning and tool invocation 
workflow.add_node("agent_reasoning", agent_reasoning)
workflow.add_node("call_tool", tool_node)

# Define the flow
workflow.add_edge(START, "agent_reasoning")

# Conditional edges: continue to tool call or end the process
workflow.add_conditional_edges(
    "agent_reasoning", should_continue, {
        "continue": "call_tool",
        "end": END
    }
)

# Normal edge: after invoking the tool, return to agent reasoning
workflow.add_edge("call_tool", "agent_reasoning")  # <-- Fixed typo here

# Set up memory for the breakpoints and add a breakpoint before calling the tool
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["call_tool"])

# Simulate user input for stock symbol
initial_input = {"messages": [{"role": "user", "content": "Should I buy AAPL stock today?"}]}
thread = {"configurable": {"thread_id": "1"}}

# Run the agent reasoning step first
for event in app.stream(initial_input, thread, stream_mode="values"):
    print(event)

# Pausing for human approval before retrieving stock price
user_approval = input("Do you approve querying the stock price for AAPL? (yes/no): ")
if user_approval.lower() == "yes":
    # Continue with tool invocation to get stock price
    for event in app.stream(None, thread, stream_mode="values"):
        print(event)
else:
    print("Execution halted by user.")

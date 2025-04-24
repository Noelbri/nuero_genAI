from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from typing import TypedDict
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Define the state for product recommendation
class RecommendationState(TypedDict):
    user_id: str  # User identifier
    preference: str  # User's current preference (e.g, genre, category)
    reasoning: str  # Reasoning process from LLM
    recommendation: str  # Final product recommendation
    memory: dict  # User memory to store preferences

# Tool function: Recommend a product based on the user's preference
@tool
def recommend_product(preference: str) -> str:
    """Recommend a product based on the user's preferences."""
    product_db = {
        "science": "I recommend 'A Brief History of Time' by Stephen Hawking.",
        "technology": "I recommend 'The Innovators' by Walter Isaacson.",
        "fiction": "I recommend 'The Alchemist' by Paulo Coelho."
    }
    # Debug: Check what product recommendation we are giving
    print(f"Recommending product for preference: {preference}")
    return product_db.get(preference, "I recommend exploring our latest products!")

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
llm = llm.bind_tools([recommend_product])

# Tool function: Update the user's memory with the latest preference
def update_memory(state: RecommendationState):
    # Store the user's preference in the memory
    state["memory"][state["user_id"]] = state["preference"]
    print(f"Updated memory: {state['memory']}")  # Debugging the memory
    return state

# Tool node to handle product recommendation
def tool_node(state: RecommendationState):
    tool_result = recommend_product.invoke({"preference": state["preference"]})
    print("Product recommendation result from tool:", tool_result)  # Debugging tool result
    state["recommendation"] = tool_result
    return state

# LLM reasoning node to process user input and generate product recommendations
def call_model(state: RecommendationState, config: RunnableConfig):
    system_prompt = SystemMessage(
        content=f"You are a helpful assistant. Recommend a product based on the user's preference for {state['preference']}."
    )
    user_prompt = HumanMessage(
        content=f"My preference is {state['preference']}."
    )

    # Send messages to the LLM
    print("Sending prompts to LLM...")
    response = llm.invoke([system_prompt, user_prompt], config)

    # Debugging the LLM response
    print("LLM Response:", response.content)

    # If the response is empty or invalid, we return a default message
    if not response.content:
        response.content = "I was unable to generate a recommendation."

    state["reasoning"] = response.content

    return state

# Conditional function to determine whether to call the tool or end
def should_continue(state: RecommendationState):
    last_message = state["reasoning"]
    print(f"Last reasoning message: {last_message}")  # Debugging the last message
    if "recommend a product" in last_message:
        return "continue"
    else:
        return "end"

# Define the workflow graph
workflow = StateGraph(RecommendationState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("update_memory", update_memory)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "update_memory")
workflow.add_edge("update_memory", "agent")

# Compile the graph
graph = workflow.compile()

# Get user input for product preference
user_preference = input("What type of product are you interested in (e.g., science, technology, fiction)? ")

# Initialize the agent's state with the user's preference and memory
initial_state = {
    "user_id": "user123",
    "preference": user_preference,
    "reasoning": "",
    "recommendation": "",
    "memory": {}
}

# Run the agent
result = graph.invoke(initial_state)

# Display the final result
print(f"\nReasoning: {result['reasoning']}")
print(f"Product Recommendation: {result['recommendation']}")
print(f"Updated Memory: {result['memory']}")

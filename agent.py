"""
Two-Stage Agent Review Workflow Implementation using LangGraph

This module implements a generic two-stage workflow where:
1. A react agent generates initial output
2. A review agent evaluates the output
3. If approved, the workflow ends; otherwise, it loops back to the react agent
"""

from typing import TypedDict, List, Annotated
from langgraph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class AgentState(TypedDict):
    """
    State schema for the two-stage agent review workflow.
    
    Attributes:
        messages: List of messages in the conversation
        current_output: Current output from the react agent
        review_count: Number of review iterations performed
        max_reviews: Maximum number of reviews allowed to prevent infinite loops
    """
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    current_output: Annotated[str, "Current output from the react agent"]
    review_count: Annotated[int, "Number of review iterations performed"]
    max_reviews: Annotated[int, "Maximum number of reviews allowed"]


# Initialize the StateGraph with our custom state
graph = StateGraph(AgentState)

# Default values for state initialization
DEFAULT_MAX_REVIEWS = 3

def initialize_state(messages: List[BaseMessage], max_reviews: int = DEFAULT_MAX_REVIEWS) -> AgentState:
    """
    Initialize the agent state with default values.
    
    Args:
        messages: Initial messages for the conversation
        max_reviews: Maximum number of review iterations (default: 3)
    
    Returns:
        AgentState: Initialized state dictionary
    """
    return {
        "messages": messages,
        "current_output": "",
        "review_count": 0,
        "max_reviews": max_reviews
    }

def react_agent(state: AgentState) -> AgentState:
    """
    React agent node function that processes messages and generates initial output.
    
    This is a dummy implementation that creates a simple response based on the input.
    In a real implementation, this would be replaced with an actual react agent
    (e.g., using LangChain's ReAct agent or similar).
    
    Args:
        state: Current agent state
        
    Returns:
        AgentState: Updated state with new output in current_output field
    """
    messages = state["messages"]
    
    # Extract the latest human message for processing
    latest_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_message = msg
            break
    
    if latest_message is None:
        # Fallback if no human message found
        response = "I don't see any input to process. Please provide a question or task."
    else:
        # Dummy logic: Create a simple response based on the input
        user_input = latest_message.content.lower()
        
        if "hello" in user_input or "hi" in user_input:
            response = "Hello! I'm a react agent. How can I help you today?"
        elif "question" in user_input or "?" in user_input:
            response = f"I understand you have a question: '{latest_message.content}'. Let me think about this step by step:\n\n1. First, I'll analyze your question\n2. Then I'll consider possible approaches\n3. Finally, I'll provide a comprehensive answer\n\nBased on my analysis, here's my response: This appears to be a thoughtful question that requires careful consideration of multiple factors."
        elif "task" in user_input or "do" in user_input:
            response = f"I see you want me to work on: '{latest_message.content}'. Here's my approach:\n\n1. Break down the task into smaller steps\n2. Identify the key requirements\n3. Execute the plan systematically\n\nMy initial solution: I'll tackle this by focusing on the core requirements and building a structured approach to address your needs."
        else:
            # Generic response for other inputs
            response = f"Thank you for your input: '{latest_message.content}'. After processing your message, here's my response:\n\nI've analyzed your request and here's what I understand: You're looking for assistance with this topic. Let me provide a comprehensive response that addresses the key points and offers practical insights."
    
    # Update the state with the generated output
    updated_state = state.copy()
    updated_state["current_output"] = response
    
    return updated_state


# Placeholder for the compiled app - will be set after building the workflow
app = None



"""
Two-Stage Agent Review Workflow using LangGraph

This module implements a generic two-stage workflow where:
1. A react agent generates initial output
2. A review agent evaluates the output
3. If approved, the workflow ends; otherwise, it loops back to the react agent

The workflow includes safeguards to prevent infinite loops.
"""

from typing import List, TypedDict, Literal
from langgraph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class AgentState(TypedDict):
    """
    State schema for the two-stage agent review workflow.
    
    Fields:
    - messages: List of conversation messages
    - current_output: Current output from the react agent
    - review_count: Number of review iterations performed
    - max_reviews: Maximum number of reviews allowed to prevent infinite loops
    """
    messages: List[BaseMessage]
    current_output: str
    review_count: int
    max_reviews: int


# Initialize the StateGraph with our custom state
graph = StateGraph(AgentState)

# Default values for state initialization
DEFAULT_MAX_REVIEWS = 3


def react_agent(state: AgentState) -> AgentState:
    """
    React agent node that generates an initial response based on the input messages.
    
    This is a dummy implementation that creates a simple response based on the input.
    In a real implementation, this would be replaced with an actual react agent
    that can perform reasoning and tool usage.
    
    Args:
        state: Current agent state containing messages and workflow tracking
        
    Returns:
        Updated state with new current_output and potentially updated messages
    """
    messages = state["messages"]
    
    # Get the latest human message to respond to
    latest_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_message = msg
            break
    
    if latest_message is None:
        # Fallback if no human message found
        response = "I don't see any input to respond to. Please provide a question or request."
    else:
        # Dummy logic: Create a simple response based on the input
        user_input = latest_message.content.lower()
        
        if "hello" in user_input or "hi" in user_input:
            response = "Hello! I'm a react agent. How can I help you today?"
        elif "question" in user_input or "?" in user_input:
            response = f"I understand you're asking: '{latest_message.content}'. Let me think about this and provide a thoughtful response based on my analysis."
        elif "help" in user_input:
            response = "I'm here to help! I can assist with various tasks including analysis, problem-solving, and providing information."
        elif "analyze" in user_input or "analysis" in user_input:
            response = f"Based on my analysis of your request '{latest_message.content}', I can provide insights and recommendations. Here's my initial assessment..."
        else:
            # Generic response for other inputs
            response = f"I've processed your input: '{latest_message.content}'. Here's my response based on my reasoning and analysis of the situation."
    
    # Add some variation to make it more realistic
    if state.get("review_count", 0) > 0:
        response += f" (Revision #{state['review_count']} - incorporating previous feedback)"
    
    # Update the state with the new output
    updated_state = state.copy()
    updated_state["current_output"] = response
    
    # Add the agent's response to the messages
    updated_state["messages"] = messages + [AIMessage(content=response)]
    
    return updated_state



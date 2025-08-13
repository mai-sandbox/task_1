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

# Placeholder for the compiled app - will be set after building the workflow
app = None

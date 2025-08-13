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


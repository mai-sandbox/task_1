"""
Two-Stage Agent Review Workflow using LangGraph

This module implements a generic two-stage workflow where:
1. A react agent generates initial output
2. A review agent evaluates the output
3. If review is good or max reviews reached, workflow ends
4. If review suggests improvement, it routes back to react agent
"""

from typing import TypedDict, List, Annotated
from langgraph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator


class AgentState(TypedDict):
    """
    State schema for the two-stage agent review workflow.
    
    Attributes:
        messages: List of messages in the conversation
        current_output: Current output from the react agent
        review_count: Number of review iterations performed
        max_reviews: Maximum number of reviews allowed (prevents infinite loops)
        review_passed: Boolean indicating if the review passed
    """
    messages: Annotated[List[BaseMessage], operator.add]
    current_output: str
    review_count: int
    max_reviews: int
    review_passed: bool


def react_agent(state: AgentState) -> AgentState:
    """
    React agent node that generates initial output based on input messages.
    
    This is a dummy implementation that creates a simple response based on the input.
    In a real implementation, this would be replaced with an actual react agent.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with generated output
    """
    # Get the latest human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if human_messages:
        latest_message = human_messages[-1].content
        
        # Dummy implementation: create a simple response
        if state["review_count"] == 0:
            # First attempt
            response = f"Initial response to: {latest_message}"
        else:
            # Revised attempt after review
            response = f"Revised response (attempt {state['review_count'] + 1}) to: {latest_message}"
            
        # Add some variation based on review count
        if state["review_count"] > 0:
            response += f" [Improved based on review feedback]"
    else:
        response = "No input message found"
    
    return {
        **state,
        "current_output": response,
        "messages": [AIMessage(content=response)]
    }


def review_agent(state: AgentState) -> AgentState:
    """
    Review agent node that evaluates the current output quality.
    
    This is a dummy implementation that uses simple heuristics to determine
    if the output is good enough. In a real implementation, this would use
    more sophisticated evaluation criteria.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with review results
    """
    current_output = state["current_output"]
    review_count = state["review_count"]
    
    # Dummy review logic - simple heuristics
    review_passed = False
    
    # Check if output meets basic criteria
    if len(current_output) > 20:  # Minimum length requirement
        # Pass review if it's the second attempt or if it contains "Improved"
        if review_count >= 1 or "Improved" in current_output:
            review_passed = True
        # Also pass if output is reasonably long and descriptive
        elif len(current_output) > 50 and any(word in current_output.lower() 
                                            for word in ["response", "answer", "solution"]):
            review_passed = True
    
    # Increment review count
    new_review_count = review_count + 1
    
    # Add review message to conversation
    review_message = f"Review {new_review_count}: {'PASSED' if review_passed else 'NEEDS_IMPROVEMENT'}"
    
    return {
        **state,
        "review_count": new_review_count,
        "review_passed": review_passed,
        "messages": [AIMessage(content=review_message)]
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional routing function that determines the next step in the workflow.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name: "react_agent" for improvement, END for completion
    """
    # End if review passed or max reviews reached
    if state["review_passed"] or state["review_count"] >= state["max_reviews"]:
        return END
    else:
        # Continue with another iteration
        return "react_agent"


# Build the StateGraph
def create_workflow() -> StateGraph:
    """
    Creates and configures the two-stage agent review workflow.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("react_agent", react_agent)
    workflow.add_node("review_agent", review_agent)
    
    # Add edges
    workflow.add_edge(START, "react_agent")
    workflow.add_edge("react_agent", "review_agent")
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "react_agent": "react_agent",
            END: END
        }
    )
    
    return workflow


# Create and compile the workflow
workflow = create_workflow()
app = workflow.compile()

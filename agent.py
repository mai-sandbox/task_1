"""
Two-Stage Agent Review Workflow Implementation

This module implements a generic two-stage workflow where:
1. A React agent processes the initial request
2. A review agent evaluates the output
3. If approved, the workflow ends; if not, it loops back to the React agent

The implementation is designed to be generic and can work with any React agent.
"""

from typing import TypedDict, List, Literal, Optional
import random
from langgraph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class AgentState(TypedDict):
    """
    State schema for the two-stage agent review workflow.
    
    Attributes:
        messages: List of conversation messages (required for LangGraph compatibility)
        current_output: Stores the React agent's current output
        review_result: Stores the review agent's decision ('approved' or 'needs_revision')
        iteration_count: Tracks the number of workflow iterations
    """
    messages: List[BaseMessage]
    current_output: Optional[str]
    review_result: Optional[Literal['approved', 'needs_revision']]
    iteration_count: int


def react_agent(state: AgentState) -> AgentState:
    """
    Dummy React agent implementation that processes user queries.
    
    This function serves as a placeholder for any actual React agent implementation.
    It extracts the user's query from the messages and generates a simulated response.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with the React agent's output
    """
    # Extract the latest human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        query = "No query provided"
    else:
        query = human_messages[-1].content
    
    # Simulate React agent processing
    # In a real implementation, this would be replaced with actual React agent logic
    response = f"React Agent Response: I have processed your query '{query}' and generated a comprehensive response with analysis and recommendations."
    
    # Update state
    updated_state = state.copy()
    updated_state["current_output"] = response
    updated_state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    # Add the React agent's response to messages
    updated_state["messages"] = state["messages"] + [AIMessage(content=response)]
    
    return updated_state


def review_agent(state: AgentState) -> AgentState:
    """
    Review agent that evaluates the React agent's output.
    
    Uses simple heuristics to determine if the output is acceptable:
    - Checks output length (longer outputs are more likely to be approved)
    - Adds some randomness to simulate real review variability
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with review decision
    """
    current_output = state.get("current_output", "")
    
    # Simple review logic based on output length and randomness
    # In a real implementation, this could use LLM-based evaluation
    output_length = len(current_output)
    
    # Longer outputs have higher chance of approval
    approval_probability = min(0.8, max(0.2, output_length / 200))
    
    # Add some randomness
    is_approved = random.random() < approval_probability
    
    if is_approved:
        review_result = "approved"
        review_message = f"Review Agent: Output approved. The response is comprehensive and addresses the query effectively. (Length: {output_length} chars)"
    else:
        review_result = "needs_revision"
        review_message = f"Review Agent: Output needs revision. Please provide more detail or improve the response quality. (Length: {output_length} chars)"
    
    # Update state
    updated_state = state.copy()
    updated_state["review_result"] = review_result
    
    # Add review message to conversation
    updated_state["messages"] = state["messages"] + [SystemMessage(content=review_message)]
    
    return updated_state


def should_continue(state: AgentState) -> Literal["react_agent", "__end__"]:
    """
    Conditional routing function that determines the next step in the workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node to execute or END
    """
    review_result = state.get("review_result")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = 3
    
    # If approved, end the workflow
    if review_result == "approved":
        return "__end__"
    
    # If needs revision but under max iterations, continue with react_agent
    if review_result == "needs_revision" and iteration_count < max_iterations:
        return "react_agent"
    
    # If max iterations reached, end with failure message
    if iteration_count >= max_iterations:
        # Add failure message to state
        failure_message = f"Workflow terminated: Maximum iterations ({max_iterations}) reached without approval."
        state["messages"].append(SystemMessage(content=failure_message))
        return "__end__"
    
    # Default case - should not reach here
    return "__end__"


# Build the StateGraph workflow
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("react_agent", react_agent)
graph.add_node("review_agent", review_agent)

# Set entry point
graph.set_entry_point("react_agent")

# Add edges
graph.add_edge("react_agent", "review_agent")
graph.add_conditional_edges(
    "review_agent",
    should_continue,
    {
        "react_agent": "react_agent",
        "__end__": END
    }
)

# Compile the graph for export
app = graph.compile()

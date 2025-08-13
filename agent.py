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


def review_agent(state: AgentState) -> AgentState:
    """
    Review agent node that evaluates the current output from the react agent.
    
    This is a dummy implementation that uses simple heuristics to decide approval.
    In a real implementation, this would be replaced with sophisticated evaluation
    criteria, potentially using another LLM or specific quality metrics.
    
    Args:
        state: Current agent state containing current_output to review
        
    Returns:
        Updated state with review message added and review_count incremented
    """
    import random
    
    current_output = state.get("current_output", "")
    review_count = state.get("review_count", 0)
    max_reviews = state.get("max_reviews", DEFAULT_MAX_REVIEWS)
    
    # Dummy evaluation logic based on simple heuristics
    approval_score = 0
    feedback_points = []
    
    # Check length - prefer responses that are not too short or too long
    if len(current_output) < 20:
        feedback_points.append("Response is too brief and lacks detail")
    elif len(current_output) > 500:
        feedback_points.append("Response is too verbose")
        approval_score += 1  # Still okay, just verbose
    else:
        approval_score += 2  # Good length
    
    # Check for key quality indicators
    if "analysis" in current_output.lower() or "think" in current_output.lower():
        approval_score += 2
        feedback_points.append("Good: Shows analytical thinking")
    
    if "help" in current_output.lower() or "assist" in current_output.lower():
        approval_score += 1
        feedback_points.append("Good: Demonstrates helpfulness")
    
    # Check for engagement
    if "?" in current_output:
        approval_score += 1
        feedback_points.append("Good: Engages with questions")
    
    # Check for politeness
    if any(word in current_output.lower() for word in ["please", "thank", "sorry"]):
        approval_score += 1
        feedback_points.append("Good: Polite tone")
    
    # Add some randomness to make it more realistic (but weighted towards approval)
    random_factor = random.choice([0, 1, 1, 2])  # Weighted towards positive
    approval_score += random_factor
    
    # Determine approval based on score and review count
    # Be more lenient as review count increases to avoid infinite loops
    approval_threshold = max(3 - review_count, 1)  # Lower threshold with more reviews
    
    # Force approval if we've reached max reviews
    if review_count >= max_reviews:
        is_approved = True
        review_message = f"APPROVED (Max reviews reached): After {review_count} reviews, accepting current output to prevent infinite loop."
    elif approval_score >= approval_threshold:
        is_approved = True
        positive_feedback = [fp for fp in feedback_points if fp.startswith("Good:")]
        if positive_feedback:
            review_message = f"APPROVED: {' '.join(positive_feedback)} Score: {approval_score}/{approval_threshold}"
        else:
            review_message = f"APPROVED: Output meets quality standards. Score: {approval_score}/{approval_threshold}"
    else:
        is_approved = False
        if feedback_points:
            review_message = f"NEEDS REVISION: {' '.join(feedback_points)} Score: {approval_score}/{approval_threshold}. Please improve the response."
        else:
            review_message = f"NEEDS REVISION: Output quality could be improved. Score: {approval_score}/{approval_threshold}. Please provide more detail and engagement."
    
    # Update the state
    updated_state = state.copy()
    updated_state["review_count"] = review_count + 1
    
    # Add review message to the conversation
    review_msg = AIMessage(
        content=review_message,
        additional_kwargs={"review_decision": "approved" if is_approved else "rejected"}
    )
    updated_state["messages"] = state["messages"] + [review_msg]
    
    return updated_state


def should_continue_review(state: AgentState) -> Literal["END", "react_agent"]:
    """
    Conditional function that determines the next step in the workflow.
    
    Checks the latest review message to determine if the output was approved
    or needs revision. Also enforces max_reviews limit to prevent infinite loops.
    
    Args:
        state: Current agent state containing messages and review tracking
        
    Returns:
        "END" if output is approved or max reviews reached
        "react_agent" if output needs revision and under review limit
    """
    messages = state.get("messages", [])
    review_count = state.get("review_count", 0)
    max_reviews = state.get("max_reviews", DEFAULT_MAX_REVIEWS)
    
    # Safety check: if we've reached max reviews, always end
    if review_count >= max_reviews:
        return "END"
    
    # Find the latest review message (should be the most recent AIMessage with review_decision)
    latest_review = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'additional_kwargs'):
            if msg.additional_kwargs and "review_decision" in msg.additional_kwargs:
                latest_review = msg
                break
    
    # If no review message found, something went wrong - end to be safe
    if latest_review is None:
        return "END"
    
    # Check the review decision
    review_decision = latest_review.additional_kwargs.get("review_decision", "")
    
    if review_decision == "approved":
        return "END"
    elif review_decision == "rejected":
        # Only continue if we haven't reached max reviews
        if review_count < max_reviews:
            return "react_agent"
        else:
            return "END"
    else:
        # Unknown decision - end to be safe
        return "END"





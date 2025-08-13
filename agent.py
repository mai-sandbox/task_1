"""
Two-Stage Agent Review Workflow using LangGraph

This module implements a generic two-stage workflow where:
1. A react agent generates an initial response
2. A review agent evaluates the output
3. If approved, the workflow finishes; otherwise, it loops back to the react agent

The workflow is designed to be generic and work with arbitrary react agents.
"""

from typing import List, Dict, Any, TypedDict, Annotated
from langgraph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import random


class AgentState(TypedDict):
    """
    State schema for the two-stage agent review workflow.
    
    Fields:
    - messages: List of conversation messages
    - current_output: Current output from the react agent
    - review_count: Number of review iterations performed
    - max_reviews: Maximum number of review attempts allowed
    """
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    current_output: Annotated[str, "Current output from the react agent"]
    review_count: Annotated[int, "Number of review iterations performed"]
    max_reviews: Annotated[int, "Maximum number of review attempts allowed"]


def react_agent(state: AgentState) -> Dict[str, Any]:
    """
    React agent node that generates an initial response based on the input messages.
    
    This is a dummy implementation that creates a simple response based on the input.
    In a real implementation, this would be replaced with an actual react agent.
    
    Args:
        state: Current agent state containing messages and other workflow data
        
    Returns:
        Dictionary with updated state fields
    """
    messages = state["messages"]
    
    # Get the latest human message to respond to
    latest_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break
    
    if not latest_message:
        latest_message = "No input provided"
    
    # Dummy implementation: Create a simple response based on the input
    # This simulates what a react agent might do - analyze the input and generate a response
    input_length = len(latest_message)
    word_count = len(latest_message.split())
    
    # Generate different types of responses based on input characteristics
    if "question" in latest_message.lower() or "?" in latest_message:
        response = f"Based on your question about '{latest_message[:50]}...', here's my analysis: This appears to be an inquiry that requires careful consideration. The question contains {word_count} words and suggests you're looking for detailed information."
    elif "help" in latest_message.lower():
        response = f"I understand you need help with: '{latest_message[:50]}...'. Let me provide assistance by breaking down the key components and offering a structured approach to address your needs."
    elif input_length > 100:
        response = f"You've provided a detailed input ({input_length} characters, {word_count} words). After analyzing your comprehensive message, I can see several key points that need addressing. Let me provide a thorough response that covers the main aspects you've mentioned."
    else:
        response = f"Thank you for your input: '{latest_message}'. I've processed your message and here's my response: This is a straightforward request that I can address directly with the following information and recommendations."
    
    # Add some variability to make responses more realistic
    response_variations = [
        f"{response} Additionally, I should mention that this type of request often benefits from a multi-step approach.",
        f"{response} It's worth noting that there are several factors to consider in this context.",
        f"{response} Based on best practices, I recommend we also consider the broader implications.",
    ]
    
    final_response = random.choice(response_variations)
    
    # Update the state with the generated response
    return {
        "current_output": final_response,
        "messages": messages + [AIMessage(content=f"React Agent Response: {final_response}")],
    }


def review_agent(state: AgentState) -> Dict[str, Any]:
    """
    Review agent node that evaluates the current output from the react agent.
    
    This is a dummy implementation that uses simple heuristics to approve or reject
    the output. In a real implementation, this would be replaced with an actual
    review agent that uses more sophisticated evaluation criteria.
    
    Args:
        state: Current agent state containing the output to review
        
    Returns:
        Dictionary with updated state fields including review decision
    """
    current_output = state.get("current_output", "")
    messages = state["messages"]
    review_count = state.get("review_count", 0)
    max_reviews = state.get("max_reviews", 3)
    
    # Increment review count
    new_review_count = review_count + 1
    
    # Dummy implementation: Use simple heuristics to evaluate quality
    approval_score = 0
    feedback_points = []
    
    # Heuristic 1: Length check (good responses should be substantial)
    if len(current_output) >= 100:
        approval_score += 2
        feedback_points.append("✓ Response has good length and detail")
    else:
        approval_score -= 1
        feedback_points.append("✗ Response seems too brief")
    
    # Heuristic 2: Check for key phrases that indicate thoughtful response
    thoughtful_phrases = [
        "analysis", "consider", "recommend", "approach", "factors",
        "implications", "best practices", "comprehensive", "detailed"
    ]
    found_phrases = [phrase for phrase in thoughtful_phrases if phrase in current_output.lower()]
    if len(found_phrases) >= 2:
        approval_score += 2
        feedback_points.append(f"✓ Contains thoughtful language: {', '.join(found_phrases[:3])}")
    else:
        approval_score -= 1
        feedback_points.append("✗ Could use more analytical language")
    
    # Heuristic 3: Check for structure and completeness
    if "." in current_output and len(current_output.split(".")) >= 2:
        approval_score += 1
        feedback_points.append("✓ Response has good sentence structure")
    else:
        feedback_points.append("✗ Response could be better structured")
    
    # Heuristic 4: Random factor to simulate subjective review (20% chance of random approval/rejection)
    random_factor = random.random()
    if random_factor < 0.1:  # 10% chance of random rejection
        approval_score -= 2
        feedback_points.append("✗ Random quality concern detected")
    elif random_factor > 0.9:  # 10% chance of random approval boost
        approval_score += 1
        feedback_points.append("✓ Exceptional quality detected")
    
    # Determine approval based on score and review count
    is_approved = approval_score >= 2
    
    # Force approval if we've reached max reviews to prevent infinite loops
    if new_review_count >= max_reviews:
        is_approved = True
        feedback_points.append(f"✓ Approved after {max_reviews} review attempts (max reached)")
    
    # Create review message
    if is_approved:
        review_status = "APPROVED"
        review_message = f"Review Agent Decision: {review_status}\n"
        review_message += f"Score: {approval_score}/5\n"
        review_message += f"Review #{new_review_count}: The output meets quality standards.\n"
        review_message += "Feedback:\n" + "\n".join(feedback_points)
    else:
        review_status = "REJECTED"
        review_message = f"Review Agent Decision: {review_status}\n"
        review_message += f"Score: {approval_score}/5\n"
        review_message += f"Review #{new_review_count}: The output needs improvement.\n"
        review_message += "Feedback for improvement:\n" + "\n".join(feedback_points)
        review_message += "\nPlease revise and resubmit."
    
    # Add review message to conversation
    updated_messages = messages + [SystemMessage(content=review_message)]
    
    # Return updated state
    return {
        "messages": updated_messages,
        "review_count": new_review_count,
        "approved": is_approved,  # Add approval status for conditional logic
    }


def should_continue_review(state: AgentState) -> str:
    """
    Conditional function that determines the next step in the workflow.
    
    This function checks if the review agent approved the output or if the maximum
    number of review attempts has been reached. It returns the next node to execute
    or 'finish' to end the workflow.
    
    Args:
        state: Current agent state containing review results and counters
        
    Returns:
        str: Next node to execute ('react_agent', 'finish')
    """
    # Check if the output was approved by the review agent
    is_approved = state.get("approved", False)
    
    # Get review count and max reviews
    review_count = state.get("review_count", 0)
    max_reviews = state.get("max_reviews", 3)
    
    # If approved, finish the workflow
    if is_approved:
        return "finish"
    
    # If we've reached the maximum number of reviews, finish anyway to prevent infinite loops
    if review_count >= max_reviews:
        return "finish"
    
    # Otherwise, send back to react_agent for another attempt
    return "react_agent"


# Initialize the StateGraph with the AgentState schema
graph = StateGraph(AgentState)

# Placeholder for the compiled graph - will be implemented in subsequent tasks
app = None





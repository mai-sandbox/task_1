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


def review_agent(state: AgentState) -> AgentState:
    """
    Review agent node function that evaluates the current output from the react agent.
    
    This is a dummy implementation that uses simple heuristics to decide approval.
    In a real implementation, this could use LLM-based evaluation, rule-based systems,
    or other quality assessment methods.
    
    Args:
        state: Current agent state containing the output to review
        
    Returns:
        AgentState: Updated state with review message added to messages list
    """
    import random
    
    current_output = state["current_output"]
    review_count = state["review_count"]
    
    # Dummy quality criteria evaluation using simple heuristics
    quality_score = 0
    feedback_points = []
    
    # Check length (longer responses generally get higher scores)
    if len(current_output) > 200:
        quality_score += 30
        feedback_points.append("Good detail and comprehensiveness")
    elif len(current_output) > 100:
        quality_score += 20
        feedback_points.append("Adequate length")
    else:
        quality_score += 5
        feedback_points.append("Response could be more detailed")
    
    # Check for structured content (numbered lists, bullet points)
    if any(marker in current_output for marker in ["1.", "2.", "3.", "•", "-", "*"]):
        quality_score += 25
        feedback_points.append("Well-structured with clear organization")
    else:
        feedback_points.append("Could benefit from better structure")
    
    # Check for key phrases that indicate thoughtful responses
    thoughtful_phrases = ["step by step", "analyze", "consider", "approach", "comprehensive"]
    if any(phrase in current_output.lower() for phrase in thoughtful_phrases):
        quality_score += 20
        feedback_points.append("Shows analytical thinking")
    else:
        feedback_points.append("Could demonstrate more analytical depth")
    
    # Check for specific examples or explanations
    if "example" in current_output.lower() or "for instance" in current_output.lower():
        quality_score += 15
        feedback_points.append("Includes helpful examples")
    
    # Add some randomness to make it more realistic (but weighted toward approval after first review)
    random_factor = random.randint(-10, 15)
    if review_count > 0:  # Be more lenient on subsequent reviews
        random_factor += 10
    
    quality_score += random_factor
    
    # Determine approval based on quality score
    approval_threshold = 60
    is_approved = quality_score >= approval_threshold
    
    # Create review message
    if is_approved:
        review_message = AIMessage(
            content=f"✅ **REVIEW APPROVED** (Score: {quality_score}/100)\n\n"
                   f"The output meets quality standards. Positive aspects:\n"
                   f"• {chr(10).join(f'  - {point}' for point in feedback_points)}\n\n"
                   f"The response is ready for delivery."
        )
    else:
        review_message = AIMessage(
            content=f"❌ **REVIEW REJECTED** (Score: {quality_score}/100)\n\n"
                   f"The output needs improvement. Feedback:\n"
                   f"• {chr(10).join(f'  - {point}' for point in feedback_points)}\n\n"
                   f"Please revise the response to address these concerns:\n"
                   f"• Provide more comprehensive details\n"
                   f"• Improve structure and organization\n"
                   f"• Include more analytical depth\n"
                   f"• Consider adding relevant examples"
        )
    
    # Update state with review message and increment review count
    updated_state = state.copy()
    updated_state["messages"] = state["messages"] + [review_message]
    updated_state["review_count"] = review_count + 1
    
    return updated_state


def should_continue_review(state: AgentState) -> str:
    """
    Conditional function that determines the next step in the workflow based on review outcome.
    
    This function checks the latest review message to determine if the output was approved
    or needs revision. It also enforces the max_reviews limit to prevent infinite loops.
    
    Args:
        state: Current agent state containing messages and review count
        
    Returns:
        str: Next node to execute - either 'END' (approved/max reviews reached) or 'react_agent' (needs revision)
    """
    messages = state["messages"]
    review_count = state["review_count"]
    max_reviews = state["max_reviews"]
    
    # Check if we've reached the maximum number of reviews
    if review_count >= max_reviews:
        return "END"  # Force end to prevent infinite loops
    
    # Find the latest review message (should be an AIMessage from review_agent)
    latest_review = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and ("REVIEW APPROVED" in msg.content or "REVIEW REJECTED" in msg.content):
            latest_review = msg
            break
    
    # If no review message found, something went wrong - end the workflow
    if latest_review is None:
        return "END"
    
    # Check if the review was approved
    if "REVIEW APPROVED" in latest_review.content:
        return "END"  # Output was approved, end the workflow
    elif "REVIEW REJECTED" in latest_review.content:
        return "react_agent"  # Output needs revision, send back to react agent
    else:
        # Fallback case - if we can't determine the review outcome, end the workflow
        return "END"


# Placeholder for the compiled app - will be set after building the workflow
app = None





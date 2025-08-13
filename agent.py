"""
Two-Stage Agent Review Workflow

This module implements a generic two-stage agent review workflow using LangGraph.
The workflow consists of:
1. A ReAct agent that generates initial output
2. A review agent that evaluates the output quality
3. Conditional routing based on review results with retry logic
"""

from typing import TypedDict, List, Literal, Optional, Any, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph import StateGraph, END
from langgraph.graph import START


class AgentState(TypedDict):
    """State schema for the two-stage agent review workflow."""
    messages: List[BaseMessage]
    current_output: str
    review_result: str
    iteration_count: int
    max_iterations: int
    react_agent: Optional[Callable]
    feedback: str


def react_agent_node(state: AgentState) -> AgentState:
    """
    Execute the ReAct agent to generate initial output.
    
    This node is designed to be generic and accept any ReAct agent implementation.
    The ReAct agent should be passed in the state or configured externally.
    """
    messages = state["messages"]
    iteration_count = state.get("iteration_count", 0)
    feedback = state.get("feedback", "")
    
    # If we have feedback from a previous review, incorporate it
    if feedback and iteration_count > 0:
        feedback_message = SystemMessage(
            content=f"Previous attempt was rejected with feedback: {feedback}. "
                   f"Please improve your response based on this feedback."
        )
        messages = messages + [feedback_message]
    
    # Check if a custom ReAct agent is provided
    react_agent = state.get("react_agent")
    
    if react_agent and callable(react_agent):
        # Use the provided ReAct agent
        try:
            response = react_agent({"messages": messages})
            if isinstance(response, dict) and "messages" in response:
                output = response["messages"][-1].content if response["messages"] else "No output generated"
            else:
                output = str(response)
        except Exception as e:
            output = f"Error executing ReAct agent: {str(e)}"
    else:
        # Default simple agent behavior for demonstration
        last_message = messages[-1].content if messages else "No input provided"
        output = f"ReAct Agent Response (Iteration {iteration_count + 1}): Processed input '{last_message}'. "
        
        if feedback:
            output += f"Addressing feedback: {feedback}. "
        
        output += "This is a simulated response that demonstrates reasoning and action capabilities."
    
    return {
        **state,
        "current_output": output,
        "iteration_count": iteration_count + 1,
        "feedback": ""  # Clear previous feedback
    }


def review_agent_node(state: AgentState) -> AgentState:
    """
    Review the output from the ReAct agent and provide feedback.
    
    This agent evaluates the quality, completeness, and correctness of the output.
    """
    current_output = state["current_output"]
    messages = state["messages"]
    iteration_count = state["iteration_count"]
    
    # Get the original user request for context
    user_request = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_request = msg.content
            break
    
    # Simple review logic - in practice, this could be a more sophisticated LLM-based reviewer
    review_criteria = [
        len(current_output) > 20,  # Minimum length check
        "error" not in current_output.lower(),  # No error messages
        current_output.strip() != "",  # Not empty
        len(current_output.split()) > 5  # Minimum word count
    ]
    
    # Additional context-aware checks
    if user_request:
        # Check if the response seems relevant to the request
        user_words = set(user_request.lower().split())
        output_words = set(current_output.lower().split())
        relevance_score = len(user_words.intersection(output_words)) / max(len(user_words), 1)
        review_criteria.append(relevance_score > 0.1)  # Some relevance required
    
    # Determine if output passes review
    passed_checks = sum(review_criteria)
    total_checks = len(review_criteria)
    quality_score = passed_checks / total_checks
    
    # Review decision logic
    if quality_score >= 0.8:  # 80% of criteria met
        review_result = "APPROVE"
        feedback = f"Output approved. Quality score: {quality_score:.2f}"
    else:
        review_result = "REJECT"
        failed_criteria = []
        
        if len(current_output) <= 20:
            failed_criteria.append("Response too short")
        if "error" in current_output.lower():
            failed_criteria.append("Contains error messages")
        if current_output.strip() == "":
            failed_criteria.append("Empty response")
        if len(current_output.split()) <= 5:
            failed_criteria.append("Insufficient detail")
        if user_request and relevance_score <= 0.1:
            failed_criteria.append("Not relevant to user request")
        
        feedback = f"Output rejected. Issues: {', '.join(failed_criteria)}. Please provide a more comprehensive and relevant response."
    
    return {
        **state,
        "review_result": review_result,
        "feedback": feedback
    }


def should_continue(state: AgentState) -> Literal["react_agent", "end"]:
    """
    Conditional routing logic based on review results and iteration limits.
    
    Routes to:
    - "react_agent": If rejected and under max iterations
    - "end": If approved or max iterations reached
    """
    review_result = state.get("review_result", "")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # If approved, end the workflow
    if review_result == "APPROVE":
        return "end"
    
    # If rejected but under max iterations, retry
    if review_result == "REJECT" and iteration_count < max_iterations:
        return "react_agent"
    
    # If max iterations reached, end the workflow
    return "end"


def create_workflow(react_agent: Optional[Callable] = None, max_iterations: int = 3) -> StateGraph:
    """
    Create the two-stage agent review workflow.
    
    Args:
        react_agent: Optional custom ReAct agent function
        max_iterations: Maximum number of retry iterations
    
    Returns:
        Compiled StateGraph workflow
    """
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    
    # Add edges
    workflow.add_edge(START, "react_agent")
    workflow.add_edge("react_agent", "review_agent")
    
    # Add conditional edge for routing
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "react_agent": "react_agent",
            "end": END
        }
    )
    
    return workflow


# Create the default workflow and compile it
workflow = create_workflow()
app = workflow.compile()


def run_workflow(
    user_input: str, 
    react_agent: Optional[Callable] = None, 
    max_iterations: int = 3
) -> dict:
    """
    Convenience function to run the workflow with a user input.
    
    Args:
        user_input: The user's input message
        react_agent: Optional custom ReAct agent
        max_iterations: Maximum retry iterations
    
    Returns:
        Final state of the workflow
    """
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "current_output": "",
        "review_result": "",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "react_agent": react_agent,
        "feedback": ""
    }
    
    result = app.invoke(initial_state)
    return result


# Export the compiled graph as required
__all__ = ["app", "run_workflow", "create_workflow", "AgentState"]

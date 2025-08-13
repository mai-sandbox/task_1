"""
Two-Stage Agent Review Workflow Implementation using LangGraph

This module implements a generic two-stage review workflow where:
1. A react agent generates initial output
2. A reviewer agent evaluates the output
3. If review passes, workflow ends; otherwise, it loops back to the react agent
"""

from typing import TypedDict, List, Literal, Callable, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    """State class that tracks the workflow progress and data."""
    messages: List[BaseMessage]
    current_output: str
    review_result: str
    iteration_count: int
    max_iterations: int


def create_two_stage_review_agent(
    react_agent: Callable[[List[BaseMessage]], str],
    reviewer_agent: Callable[[str], str] = None
) -> StateGraph:
    """
    Creates a two-stage review workflow agent that accepts an arbitrary react agent.
    
    Args:
        react_agent: A callable that takes a list of messages and returns a string output
        reviewer_agent: Optional callable that takes output string and returns review result.
                       If None, uses a default reviewer.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    
    def react_agent_node(state: AgentState) -> AgentState:
        """
        Node that runs the react agent to generate output.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with current_output from react agent
        """
        try:
            # Generate output using the provided react agent
            output = react_agent(state["messages"])
            
            return {
                **state,
                "current_output": output,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
        except Exception as e:
            # Handle errors gracefully
            return {
                **state,
                "current_output": f"Error in react agent: {str(e)}",
                "iteration_count": state.get("iteration_count", 0) + 1
            }
    
    def default_reviewer_agent(output: str) -> str:
        """
        Default reviewer agent that provides basic output evaluation.
        
        Args:
            output: The output to review
            
        Returns:
            Review result as string
        """
        # Simple heuristic-based review
        if not output or len(output.strip()) < 10:
            return "REJECT: Output too short or empty"
        
        if "error" in output.lower() or "failed" in output.lower():
            return "REJECT: Output contains error indicators"
        
        if len(output.strip()) > 50:  # Reasonable length threshold
            return "APPROVE: Output meets basic quality criteria"
        
        return "REJECT: Output does not meet quality standards"
    
    def reviewer_agent_node(state: AgentState) -> AgentState:
        """
        Node that reviews the current output and determines if it's acceptable.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with review_result
        """
        current_output = state.get("current_output", "")
        
        # Use provided reviewer or default
        reviewer = reviewer_agent if reviewer_agent else default_reviewer_agent
        
        try:
            review_result = reviewer(current_output)
            
            return {
                **state,
                "review_result": review_result
            }
        except Exception as e:
            # Handle reviewer errors
            return {
                **state,
                "review_result": f"REJECT: Reviewer error - {str(e)}"
            }
    
    def should_continue(state: AgentState) -> Literal["reviewer", "react_agent", "end"]:
        """
        Conditional routing function that determines the next step in the workflow.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node to execute: 'reviewer', 'react_agent', or 'end'
        """
        # After react agent, always go to reviewer
        if not state.get("review_result"):
            return "reviewer"
        
        # Check if review passed
        review_result = state.get("review_result", "")
        if review_result.startswith("APPROVE"):
            return "end"
        
        # Check if we've exceeded max iterations
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        if iteration_count >= max_iterations:
            return "end"
        
        # Review failed and we haven't exceeded max iterations, try again
        return "react_agent"
    
    # Create the StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_node("reviewer_agent", reviewer_agent_node)
    
    # Set entry point
    workflow.add_edge(START, "react_agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "react_agent",
        should_continue,
        {
            "reviewer": "reviewer_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "reviewer_agent",
        should_continue,
        {
            "react_agent": "react_agent",
            "end": END
        }
    )
    
    return workflow


def default_react_agent(messages: List[BaseMessage]) -> str:
    """
    Default react agent implementation for testing purposes.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Generated response as string
    """
    if not messages:
        return "No messages provided"
    
    # Get the last human message
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        content = last_message.content
        # Simple echo with some processing
        return f"Processed response to: {content}. This is a default react agent response."
    
    return "Unable to process the provided messages"


# Create the default agent with a simple react agent
app = create_two_stage_review_agent(default_react_agent).compile()


def create_agent_with_custom_react_agent(react_agent: Callable[[List[BaseMessage]], str]) -> StateGraph:
    """
    Convenience function to create an agent with a custom react agent.
    
    Args:
        react_agent: Custom react agent function
        
    Returns:
        Compiled StateGraph
    """
    return create_two_stage_review_agent(react_agent).compile()


# Example usage and testing function
def test_agent():
    """Test function to demonstrate agent usage."""
    from langchain_core.messages import HumanMessage
    
    # Test with default agent
    initial_state = {
        "messages": [HumanMessage("Hello, can you help me with a task?")],
        "current_output": "",
        "review_result": "",
        "iteration_count": 0,
        "max_iterations": 3
    }
    
    result = app.invoke(initial_state)
    print("Test Result:")
    print(f"Final Output: {result.get('current_output', 'No output')}")
    print(f"Review Result: {result.get('review_result', 'No review')}")
    print(f"Iterations: {result.get('iteration_count', 0)}")


if __name__ == "__main__":
    test_agent()

"""
Two-Stage Agent Review Workflow Implementation

This module implements a generic two-stage review workflow using LangGraph that can wrap
any ReAct agent. The workflow consists of:
1. Initial Agent: Runs the provided ReAct agent to generate initial output
2. Review Agent: Evaluates the initial output for quality/correctness
3. Conditional Routing: Routes back to initial agent if review fails, or finishes if review passes

The implementation prevents infinite loops with a maximum iteration limit.
"""

from typing import Literal, TypedDict, Annotated, Any, Dict, List
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


class ReviewWorkflowState(MessagesState):
    """
    Extended state schema for the review workflow.
    
    Inherits from MessagesState to maintain standard message handling,
    and adds fields for tracking review results and iteration count.
    """
    review_result: NotRequired[str]  # Result of the review agent's evaluation
    iteration_count: NotRequired[int]  # Track number of iterations to prevent infinite loops
    max_iterations: NotRequired[int]  # Maximum allowed iterations


def create_review_workflow(
    initial_agent: Runnable,
    review_agent: Runnable = None,
    max_iterations: int = 3
) -> StateGraph:
    """
    Create a two-stage review workflow that wraps an arbitrary ReAct agent.
    
    Args:
        initial_agent: The ReAct agent to be wrapped in the review workflow
        review_agent: Optional custom review agent. If None, creates a default one
        max_iterations: Maximum number of review iterations to prevent infinite loops
    
    Returns:
        Compiled StateGraph implementing the review workflow
    """
    
    # Create default review agent if none provided
    if review_agent is None:
        review_agent = create_react_agent(
            model="gpt-3.5-turbo",  # Default model - can be overridden
            tools=[],  # Review agent doesn't need tools by default
            prompt=(
                "You are a review agent. Your job is to evaluate the quality and correctness "
                "of responses from another agent. Analyze the response and determine if it:\n"
                "1. Adequately answers the user's question\n"
                "2. Is factually accurate (if verifiable)\n"
                "3. Is well-structured and clear\n"
                "4. Follows any specific instructions given\n\n"
                "Respond with either:\n"
                "- 'APPROVED' if the response is good and should be accepted\n"
                "- 'NEEDS_IMPROVEMENT: [specific feedback]' if the response needs to be revised\n\n"
                "Be constructive in your feedback and specific about what needs improvement."
            ),
            name="review_agent"
        )
    
    def initial_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """
        Node that runs the initial ReAct agent.
        
        Processes the current state through the provided initial agent
        and returns the updated state with the agent's response.
        """
        # Initialize iteration count if not present
        iteration_count = state.get("iteration_count", 0)
        
        # Run the initial agent
        result = initial_agent.invoke(state)
        
        # Update state with agent result and increment iteration count
        return {
            "messages": result["messages"],
            "iteration_count": iteration_count + 1,
            "max_iterations": max_iterations
        }
    
    def review_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """
        Node that runs the review agent to evaluate the initial agent's output.
        
        Takes the last message from the initial agent and evaluates it,
        storing the review result in the state.
        """
        # Get the last AI message from the initial agent
        messages = state["messages"]
        last_ai_message = None
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            return {
                "review_result": "NEEDS_IMPROVEMENT: No response found to review",
                "messages": messages + [SystemMessage(content="No AI response found to review")]
            }
        
        # Create review prompt with the response to evaluate
        review_messages = [
            HumanMessage(content=f"Please review this response: {last_ai_message.content}")
        ]
        
        # Run the review agent
        review_state = {"messages": review_messages}
        review_result = review_agent.invoke(review_state)
        
        # Extract the review decision from the review agent's response
        review_content = review_result["messages"][-1].content
        
        return {
            "review_result": review_content,
            "messages": messages + [SystemMessage(content=f"Review: {review_content}")]
        }
    
    def should_continue(state: ReviewWorkflowState) -> Literal["continue", "finish"]:
        """
        Conditional routing function that determines whether to continue the review loop
        or finish the workflow.
        
        Routes to:
        - "continue": If review failed and we haven't exceeded max iterations
        - "finish": If review passed or max iterations reached
        """
        review_result = state.get("review_result", "")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # Check if we've exceeded maximum iterations
        if iteration_count >= max_iterations:
            return "finish"
        
        # Check if review approved the response
        if review_result.startswith("APPROVED"):
            return "finish"
        
        # If review indicates improvement needed, continue the loop
        if review_result.startswith("NEEDS_IMPROVEMENT"):
            return "continue"
        
        # Default to finish if review result is unclear
        return "finish"
    
    # Build the workflow graph
    workflow = StateGraph(ReviewWorkflowState)
    
    # Add nodes
    workflow.add_node("initial_agent", initial_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    
    # Add edges
    workflow.add_edge(START, "initial_agent")
    workflow.add_edge("initial_agent", "review_agent")
    
    # Add conditional edge for the review loop
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "continue": "initial_agent",  # Loop back to initial agent
            "finish": END  # End the workflow
        }
    )
    
    return workflow


# Create a default example workflow for demonstration
def create_default_example_agent():
    """
    Create a simple example ReAct agent for demonstration purposes.
    This shows how the review workflow can wrap any ReAct agent.
    """
    return create_react_agent(
        model="gpt-3.5-turbo",
        tools=[],  # No tools for this simple example
        prompt=(
            "You are a helpful assistant. Answer the user's questions clearly and concisely. "
            "If you're unsure about something, say so rather than guessing."
        ),
        name="example_agent"
    )


# Create the main application graph
# This demonstrates the review workflow with a default agent
example_initial_agent = create_default_example_agent()
workflow = create_review_workflow(example_initial_agent)

# Compile the graph and export as 'app' (required by LangGraph conventions)
app = workflow.compile()


# Additional utility function for users who want to create their own workflow
def create_custom_review_workflow(
    initial_agent: Runnable,
    review_model: str = "gpt-3.5-turbo",
    review_prompt: str = None,
    max_iterations: int = 3
) -> StateGraph:
    """
    Utility function to create a custom review workflow with specified parameters.
    
    Args:
        initial_agent: The ReAct agent to wrap
        review_model: Model to use for the review agent
        review_prompt: Custom prompt for the review agent
        max_iterations: Maximum review iterations
    
    Returns:
        Compiled StateGraph ready for use
    """
    if review_prompt is None:
        review_prompt = (
            "You are a review agent. Evaluate responses for quality, accuracy, and completeness. "
            "Respond with 'APPROVED' if good, or 'NEEDS_IMPROVEMENT: [feedback]' if not."
        )
    
    custom_review_agent = create_react_agent(
        model=review_model,
        tools=[],
        prompt=review_prompt,
        name="custom_review_agent"
    )
    
    workflow = create_review_workflow(
        initial_agent=initial_agent,
        review_agent=custom_review_agent,
        max_iterations=max_iterations
    )
    
    return workflow.compile()


if __name__ == "__main__":
    # Example usage
    print("Two-Stage Agent Review Workflow created successfully!")
    print("The 'app' variable contains the compiled graph ready for deployment.")
    print("\nWorkflow structure:")
    print("1. User input -> Initial Agent")
    print("2. Initial Agent output -> Review Agent")
    print("3. Review Agent -> Decision (continue/finish)")
    print("4. If continue: back to Initial Agent with feedback")
    print("5. If finish: return final result")

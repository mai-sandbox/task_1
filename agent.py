"""
Review-and-Refine Agent Workflow using LangGraph

This module implements a generic Review-and-Refine workflow that can work with any ReAct agent.
The workflow follows: Initial Agent → Review Agent → Decision (finish or refine)
"""

from typing import TypedDict, List, Annotated, Any, Callable
from langgraph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import Runnable
import operator


class ReviewRefineState(TypedDict):
    """State schema for the Review-and-Refine workflow"""
    messages: Annotated[List[BaseMessage], operator.add]
    current_output: str
    review_feedback: str
    iteration_count: int
    max_iterations: int
    initial_agent: Any  # The ReAct agent to be used
    is_satisfactory: bool


def initial_agent_node(state: ReviewRefineState) -> ReviewRefineState:
    """
    Node that runs the initial ReAct agent to generate output.
    
    Args:
        state: Current workflow state containing the agent and messages
        
    Returns:
        Updated state with the agent's output
    """
    messages = state["messages"]
    initial_agent = state["initial_agent"]
    
    # If this is a refinement iteration, add the review feedback to the messages
    if state.get("review_feedback") and state["iteration_count"] > 0:
        feedback_message = HumanMessage(
            content=f"Please improve your previous response based on this feedback: {state['review_feedback']}"
        )
        messages = messages + [feedback_message]
    
    # Run the initial agent
    if hasattr(initial_agent, 'invoke'):
        # If it's a LangChain runnable
        response = initial_agent.invoke({"messages": messages})
        if hasattr(response, 'content'):
            output = response.content
        else:
            output = str(response)
    elif callable(initial_agent):
        # If it's a callable function
        output = initial_agent(messages)
    else:
        # Fallback for other agent types
        output = str(initial_agent)
    
    return {
        **state,
        "current_output": output,
        "messages": messages + [AIMessage(content=output)],
        "iteration_count": state["iteration_count"] + 1
    }


def review_agent_node(state: ReviewRefineState) -> ReviewRefineState:
    """
    Node that reviews the current output and provides feedback.
    
    Args:
        state: Current workflow state with the output to review
        
    Returns:
        Updated state with review feedback and satisfaction status
    """
    current_output = state["current_output"]
    
    # Simple review logic - in a real implementation, this would use another LLM
    # For now, we'll implement a basic review system
    review_prompt = f"""
    Please review the following output and determine if it's satisfactory:
    
    Output: {current_output}
    
    Provide feedback on:
    1. Clarity and coherence
    2. Completeness of the response
    3. Accuracy (if applicable)
    4. Overall quality
    
    If the output is good, respond with "SATISFACTORY: [brief positive feedback]"
    If it needs improvement, respond with "NEEDS_IMPROVEMENT: [specific feedback for improvement]"
    """
    
    # For demonstration purposes, we'll use a simple heuristic
    # In practice, this would be another LLM agent
    if len(current_output.strip()) < 10:
        feedback = "NEEDS_IMPROVEMENT: The response is too short and lacks detail. Please provide a more comprehensive answer."
        is_satisfactory = False
    elif "error" in current_output.lower() or "sorry" in current_output.lower():
        feedback = "NEEDS_IMPROVEMENT: The response indicates an error or inability to complete the task. Please try a different approach."
        is_satisfactory = False
    else:
        # Simple positive case - in reality this would be more sophisticated
        feedback = "SATISFACTORY: The response appears complete and well-structured."
        is_satisfactory = True
    
    return {
        **state,
        "review_feedback": feedback,
        "is_satisfactory": is_satisfactory
    }


def decision_node(state: ReviewRefineState) -> str:
    """
    Decision node that determines whether to finish or refine based on review.
    
    Args:
        state: Current workflow state with review results
        
    Returns:
        Next node name: "finish" or "refine"
    """
    # Check if we've reached max iterations
    if state["iteration_count"] >= state["max_iterations"]:
        return "finish"
    
    # Check if the review is satisfactory
    if state["is_satisfactory"]:
        return "finish"
    
    # Otherwise, refine
    return "refine"


def create_review_refine_workflow(initial_agent: Any, max_iterations: int = 3) -> StateGraph:
    """
    Creates a Review-and-Refine workflow graph.
    
    Args:
        initial_agent: The ReAct agent to use for generating initial output
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the state graph
    workflow = StateGraph(ReviewRefineState)
    
    # Add nodes
    workflow.add_node("initial_agent", initial_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    workflow.add_node("decision", decision_node)
    
    # Add edges
    workflow.add_edge(START, "initial_agent")
    workflow.add_edge("initial_agent", "review_agent")
    workflow.add_conditional_edges(
        "review_agent",
        decision_node,
        {
            "finish": END,
            "refine": "initial_agent"
        }
    )
    
    return workflow


def default_react_agent(messages: List[BaseMessage]) -> str:
    """
    Default ReAct agent implementation for demonstration purposes.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Agent response as string
    """
    if not messages:
        return "Hello! How can I help you today?"
    
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        content = last_message.content
    else:
        content = str(last_message)
    
    # Simple echo response with some processing
    return f"I understand you're asking about: {content}. Let me provide a helpful response based on that."


# Create the default workflow with a simple ReAct agent
default_workflow = create_review_refine_workflow(
    initial_agent=default_react_agent,
    max_iterations=3
)

# Compile the graph and export as 'app'
app = default_workflow.compile()


def create_custom_workflow(initial_agent: Any, max_iterations: int = 3):
    """
    Factory function to create a custom Review-and-Refine workflow with a specific agent.
    
    Args:
        initial_agent: Custom ReAct agent to use
        max_iterations: Maximum refinement iterations
        
    Returns:
        Compiled workflow ready for execution
    """
    workflow = create_review_refine_workflow(initial_agent, max_iterations)
    return workflow.compile()


# Example usage function
def run_workflow_example():
    """Example of how to use the Review-and-Refine workflow"""
    
    # Example initial state
    initial_state = {
        "messages": [HumanMessage(content="Explain quantum computing in simple terms")],
        "current_output": "",
        "review_feedback": "",
        "iteration_count": 0,
        "max_iterations": 3,
        "initial_agent": default_react_agent,
        "is_satisfactory": False
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Run example when script is executed directly
    result = run_workflow_example()
    print("Workflow completed!")
    print(f"Final output: {result.get('current_output', 'No output')}")
    print(f"Iterations: {result.get('iteration_count', 0)}")
    print(f"Review feedback: {result.get('review_feedback', 'No feedback')}")

from typing import TypedDict, Literal, Optional, Any, Callable
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
import operator
from typing import Annotated, Sequence


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int
    max_iterations: int
    current_output: Optional[str]
    review_result: Optional[str]
    is_approved: bool


def react_agent_node(state: AgentState) -> AgentState:
    """
    Generic React agent node that accepts an arbitrary react agent function.
    This function should be provided when creating the graph.
    """
    messages = state["messages"]
    
    # Get the last human message as input
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        input_text = last_message.content
    else:
        input_text = "Please provide a response."
    
    # For demo purposes, we'll create a simple response
    # In practice, this would call the provided react agent
    react_output = f"React Agent Response to: '{input_text}'\n\nThis is a placeholder response from the react agent. In a real implementation, this would be replaced with your actual react agent logic."
    
    return {
        **state,
        "current_output": react_output,
        "messages": messages + [AIMessage(content=f"React agent generated: {react_output}")]
    }


def review_agent_node(state: AgentState) -> AgentState:
    """
    Review agent that evaluates the react agent's output.
    """
    current_output = state["current_output"]
    
    # Simple review logic - in practice this could be more sophisticated
    review_prompt = f"""
    Please review the following output and determine if it's satisfactory:
    
    Output: {current_output}
    
    Is this output good enough? Reply with 'APPROVED' if yes, 'REJECTED' if no, followed by your reasoning.
    """
    
    # For demo purposes, we'll simulate a review
    # In practice, this would call an actual LLM for review
    review_result = f"APPROVED - The output appears to be a valid response that addresses the input appropriately."
    
    is_approved = review_result.startswith("APPROVED")
    
    return {
        **state,
        "review_result": review_result,
        "is_approved": is_approved,
        "messages": state["messages"] + [AIMessage(content=f"Review result: {review_result}")]
    }


def should_continue(state: AgentState) -> Literal["react_agent", "end"]:
    """
    Determines whether to continue with another iteration or end.
    """
    if state["is_approved"]:
        return "end"
    
    if state["iterations"] >= state["max_iterations"]:
        return "end"
    
    return "react_agent"


def increment_iteration(state: AgentState) -> AgentState:
    """
    Increments the iteration counter for feedback loop.
    """
    return {
        **state,
        "iterations": state["iterations"] + 1,
        "messages": state["messages"] + [AIMessage(content=f"Iteration {state['iterations'] + 1}: Sending back to react agent for improvement")]
    }


def create_react_review_agent(react_agent_func: Optional[Callable] = None, max_iterations: int = 3) -> StateGraph:
    """
    Creates a generic react-review-iterate agent that can work with any react agent.
    
    Args:
        react_agent_func: Optional function that implements the react agent logic
        max_iterations: Maximum number of iterations before stopping
    
    Returns:
        Compiled LangGraph StateGraph
    """
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    workflow.add_node("increment_iteration", increment_iteration)
    
    # Add edges
    workflow.set_entry_point("react_agent")
    workflow.add_edge("react_agent", "review_agent")
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "react_agent": "increment_iteration",
            "end": END
        }
    )
    workflow.add_edge("increment_iteration", "react_agent")
    
    return workflow


# Create and compile the default agent
workflow = create_react_review_agent()

app = workflow.compile()


def run_agent(input_message: str, max_iterations: int = 3) -> dict:
    """
    Convenience function to run the agent with a simple input.
    
    Args:
        input_message: The input message to process
        max_iterations: Maximum number of iterations
    
    Returns:
        Final state of the agent
    """
    initial_state = {
        "messages": [HumanMessage(content=input_message)],
        "iterations": 0,
        "max_iterations": max_iterations,
        "current_output": None,
        "review_result": None,
        "is_approved": False
    }
    
    return app.invoke(initial_state)
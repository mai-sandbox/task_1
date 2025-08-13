from typing import TypedDict, Literal, Any, Callable, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, MessagesState


class OrchestratorState(TypedDict):
    messages: list[BaseMessage]
    react_output: Optional[str]
    review_result: Optional[dict]
    iteration_count: int
    max_iterations: int
    react_agent: Optional[Callable]


def create_orchestrator_agent(react_agent_fn: Callable, max_iterations: int = 3):
    """
    Creates an orchestrator agent that coordinates between a ReAct agent and a review agent.
    
    Args:
        react_agent_fn: A callable that takes a state and returns the ReAct agent's response
        max_iterations: Maximum number of iterations before giving up (default: 3)
    
    Returns:
        Compiled LangGraph application
    """
    
    def run_react_agent(state: OrchestratorState) -> OrchestratorState:
        """Run the ReAct agent and store its output"""
        try:
            # Call the provided React agent function
            result = react_agent_fn(state)
            
            # Extract the output (assuming it's in the last AI message)
            react_output = result.get("messages", [])[-1].content if result.get("messages") else "No output generated"
            
            return {
                "messages": state["messages"],
                "react_output": react_output,
                "review_result": state.get("review_result"),
                "iteration_count": state["iteration_count"],
                "max_iterations": state["max_iterations"],
                "react_agent": state.get("react_agent")
            }
        except Exception as e:
            return {
                "messages": state["messages"],
                "react_output": f"Error in ReAct agent: {str(e)}",
                "review_result": state.get("review_result"),
                "iteration_count": state["iteration_count"],
                "max_iterations": state["max_iterations"],
                "react_agent": state.get("react_agent")
            }
    
    def run_review_agent(state: OrchestratorState) -> OrchestratorState:
        """Review the ReAct agent's output and decide if it's acceptable"""
        react_output = state.get("react_output", "")
        
        # Simple review logic - can be enhanced with an actual LLM reviewer
        review_result = {
            "is_good": len(react_output) > 10 and "error" not in react_output.lower(),
            "feedback": "Output looks good" if len(react_output) > 10 and "error" not in react_output.lower() 
                       else "Output needs improvement - too short or contains errors"
        }
        
        return {
            "messages": state["messages"],
            "react_output": state["react_output"],
            "review_result": review_result,
            "iteration_count": state["iteration_count"],
            "max_iterations": state["max_iterations"],
            "react_agent": state.get("react_agent")
        }
    
    def increment_iteration(state: OrchestratorState) -> OrchestratorState:
        """Increment the iteration counter"""
        return {
            "messages": state["messages"],
            "react_output": None,  # Reset for next iteration
            "review_result": None,  # Reset for next iteration
            "iteration_count": state["iteration_count"] + 1,
            "max_iterations": state["max_iterations"],
            "react_agent": state.get("react_agent")
        }
    
    def finalize_result(state: OrchestratorState) -> OrchestratorState:
        """Finalize the result and add it to messages"""
        react_output = state.get("react_output", "No output generated")
        review_result = state.get("review_result", {})
        
        final_message = AIMessage(
            content=f"Final Result (after {state['iteration_count']} iterations): {react_output}"
        )
        
        return {
            "messages": state["messages"] + [final_message],
            "react_output": state["react_output"],
            "review_result": state["review_result"],
            "iteration_count": state["iteration_count"],
            "max_iterations": state["max_iterations"],
            "react_agent": state.get("react_agent")
        }
    
    def should_continue(state: OrchestratorState) -> Literal["continue", "finish", "max_iterations"]:
        """Decide whether to continue iterating or finish"""
        review_result = state.get("review_result", {})
        
        # Check if we've hit max iterations
        if state["iteration_count"] >= state["max_iterations"]:
            return "max_iterations"
        
        # Check if the review says the output is good
        if review_result.get("is_good", False):
            return "finish"
        
        return "continue"
    
    # Build the graph
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes
    workflow.add_node("react_agent", run_react_agent)
    workflow.add_node("review_agent", run_review_agent)
    workflow.add_node("increment_iteration", increment_iteration)
    workflow.add_node("finalize_result", finalize_result)
    
    # Set entry point
    workflow.set_entry_point("react_agent")
    
    # Add edges
    workflow.add_edge("react_agent", "review_agent")
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "continue": "increment_iteration",
            "finish": "finalize_result",
            "max_iterations": "finalize_result"
        }
    )
    workflow.add_edge("increment_iteration", "react_agent")
    workflow.add_edge("finalize_result", END)
    
    return workflow.compile()


# Default simple ReAct agent for demonstration
def simple_react_agent(state: OrchestratorState) -> dict:
    """
    A simple ReAct agent implementation for demonstration.
    In practice, you would replace this with your actual ReAct agent.
    """
    messages = state.get("messages", [])
    last_human_message = None
    
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break
    
    if not last_human_message:
        response = "I need a question or task to work on."
    else:
        # Simple response - in practice this would be a sophisticated ReAct agent
        user_input = last_human_message.content
        response = f"I analyzed your request: '{user_input}'. Based on my reasoning, here's my response: This is a thoughtful analysis of the topic with relevant considerations and conclusions."
    
    return {
        "messages": messages + [AIMessage(content=response)]
    }


# Create the main app with the default agent
app = create_orchestrator_agent(simple_react_agent, max_iterations=3)


# Helper function to create app with custom ReAct agent
def create_app_with_custom_agent(react_agent_fn: Callable, max_iterations: int = 3):
    """
    Create an orchestrator app with a custom ReAct agent.
    
    Args:
        react_agent_fn: Your custom ReAct agent function
        max_iterations: Maximum iterations before stopping
    
    Returns:
        Compiled LangGraph application
    """
    return create_orchestrator_agent(react_agent_fn, max_iterations)
"""
Two-Stage Agent Review Workflow

A generic LangGraph implementation that runs a React agent to generate initial output,
then uses a review agent to evaluate that output. If the review is satisfactory, the
workflow finishes. Otherwise, it sends feedback back to the original agent for revision.

This implementation is designed to work with arbitrary React agents by accepting
configurable tools and prompts.
"""

from typing import List, Literal, Optional, Any, Dict
from typing_extensions import Annotated

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent


# Custom state schema extending MessagesState
class ReviewWorkflowState(MessagesState):
    """State for the two-stage review workflow."""
    
    # Track review status
    review_status: Annotated[Optional[str], "Current review status: 'pending', 'approved', 'needs_revision'"]
    
    # Track iteration count to prevent infinite loops
    iteration_count: Annotated[int, "Number of review iterations completed"]
    
    # Store the current task/prompt for the workflow
    current_task: Annotated[Optional[str], "The current task being worked on"]
    
    # Store review feedback for revisions
    review_feedback: Annotated[Optional[str], "Feedback from the review agent"]


def create_main_agent(model: Any, tools: List[Any], prompt: str, name: str = "main_agent"):
    """
    Create a generic React agent that can work with arbitrary tools and prompts.
    
    Args:
        model: The language model to use
        tools: List of tools available to the agent
        prompt: Custom prompt for the agent
        name: Name of the agent
    
    Returns:
        Compiled React agent
    """
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name=name
    )


def create_review_agent(model: Any, name: str = "review_agent"):
    """
    Create a review agent that evaluates outputs from the main agent.
    
    Args:
        model: The language model to use
        name: Name of the review agent
    
    Returns:
        Compiled React agent configured for review tasks
    """
    review_prompt = """You are a review agent responsible for evaluating the quality and completeness of work done by other agents.

Your task is to:
1. Carefully review the previous agent's output
2. Determine if the work meets the requirements and is of good quality
3. Provide clear feedback

EVALUATION CRITERIA:
- Is the response complete and addresses all aspects of the request?
- Is the information accurate and well-structured?
- Are there any obvious errors or omissions?
- Is the response helpful and actionable?

RESPONSE FORMAT:
You must respond with one of these exact formats:

If the work is satisfactory:
APPROVED: [Brief explanation of why the work is good]

If the work needs improvement:
NEEDS_REVISION: [Specific feedback on what needs to be improved]

Be thorough but concise in your evaluation."""

    return create_react_agent(
        model=model,
        tools=[],  # Review agent doesn't need external tools
        prompt=review_prompt,
        name=name
    )


def main_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
    """
    Node that runs the main React agent.
    
    This node is designed to be generic and work with any React agent configuration.
    """
    # Default model - can be overridden via configuration
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    # Default tools and prompt - in a real implementation, these would be configurable
    default_prompt = """You are a helpful assistant. Complete the given task to the best of your ability.
    
If this is a revision based on feedback, carefully consider the review feedback and improve your previous response accordingly.

Current task: {task}

{feedback_context}"""
    
    # Format the prompt with current context
    task = state.get("current_task", "Complete the user's request")
    feedback_context = ""
    
    if state.get("review_feedback"):
        feedback_context = f"\nPrevious review feedback to address:\n{state['review_feedback']}"
    
    formatted_prompt = default_prompt.format(task=task, feedback_context=feedback_context)
    
    # Create the main agent (in practice, this would be passed in or configured)
    main_agent = create_main_agent(
        model=model,
        tools=[],  # Default empty tools - would be configurable
        prompt=formatted_prompt
    )
    
    # Run the main agent
    result = main_agent.invoke(state)
    
    # Update state with results
    return {
        "messages": result["messages"],
        "review_status": "pending",
        "iteration_count": state.get("iteration_count", 0)
    }


def review_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
    """
    Node that runs the review agent to evaluate the main agent's output.
    """
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    # Create the review agent
    review_agent = create_review_agent(model)
    
    # Run the review agent
    result = review_agent.invoke(state)
    
    # Extract the review decision from the last message
    last_message = result["messages"][-1]
    review_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Parse the review decision
    if "APPROVED:" in review_content:
        review_status = "approved"
        review_feedback = None
    elif "NEEDS_REVISION:" in review_content:
        review_status = "needs_revision"
        # Extract feedback after "NEEDS_REVISION:"
        feedback_start = review_content.find("NEEDS_REVISION:") + len("NEEDS_REVISION:")
        review_feedback = review_content[feedback_start:].strip()
    else:
        # Fallback parsing
        review_status = "needs_revision"
        review_feedback = "Please review and improve the previous response."
    
    return {
        "messages": result["messages"],
        "review_status": review_status,
        "review_feedback": review_feedback,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def should_continue(state: ReviewWorkflowState) -> Literal["main_agent", "END"]:
    """
    Conditional routing function that determines the next step based on review results.
    
    Returns:
        - "main_agent" if the work needs revision and we haven't exceeded max iterations
        - "END" if the work is approved or we've reached the maximum iterations
    """
    # Check if we've reached the maximum number of iterations (prevent infinite loops)
    max_iterations = 5
    current_iterations = state.get("iteration_count", 0)
    
    if current_iterations >= max_iterations:
        return "END"
    
    # Check review status
    review_status = state.get("review_status")
    
    if review_status == "approved":
        return "END"
    elif review_status == "needs_revision":
        return "main_agent"
    else:
        # Default to ending if status is unclear
        return "END"


def setup_workflow_state(task: str) -> ReviewWorkflowState:
    """
    Helper function to set up initial workflow state.
    
    Args:
        task: The task description for the workflow
    
    Returns:
        Initial state for the workflow
    """
    return {
        "messages": [HumanMessage(content=task)],
        "review_status": None,
        "iteration_count": 0,
        "current_task": task,
        "review_feedback": None
    }


# Build the workflow graph
def create_review_workflow() -> StateGraph:
    """
    Create the two-stage review workflow graph.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the state graph
    workflow = StateGraph(ReviewWorkflowState)
    
    # Add nodes
    workflow.add_node("main_agent", main_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    
    # Add edges
    workflow.add_edge(START, "main_agent")
    workflow.add_edge("main_agent", "review_agent")
    
    # Add conditional edge for the review decision
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "main_agent": "main_agent",
            "END": END
        }
    )
    
    return workflow


# Create and compile the workflow
workflow = create_review_workflow()
app = workflow.compile()


# Example usage function (for testing/demonstration)
def run_review_workflow(task: str, max_iterations: int = 5) -> Dict[str, Any]:
    """
    Run the two-stage review workflow with a given task.
    
    Args:
        task: The task to be completed
        max_iterations: Maximum number of review iterations
    
    Returns:
        Final state of the workflow
    """
    initial_state = setup_workflow_state(task)
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Example usage
    test_task = "Write a brief summary of the benefits of renewable energy."
    result = run_review_workflow(test_task)
    
    print("Workflow completed!")
    print(f"Final status: {result.get('review_status')}")
    print(f"Iterations: {result.get('iteration_count')}")
    print(f"Final message: {result['messages'][-1].content if result['messages'] else 'No messages'}")


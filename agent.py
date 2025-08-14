"""
Two-Stage Agent Review Workflow

This module implements a generic two-stage review workflow using LangGraph where:
1. An initial React agent generates output
2. A review agent evaluates the output
3. If approved, the workflow ends; if not, it loops back to the initial agent

The workflow is designed to be generic and accept arbitrary React agents.
"""

from typing import Annotated, Literal, Dict, Any, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


# State Schema extending MessagesState with review fields
class ReviewWorkflowState(MessagesState):
    """
    State schema for the two-stage review workflow.
    Extends MessagesState to include review status and iteration tracking.
    """
    review_status: str = "pending"  # "pending", "approved", "needs_revision"
    iteration_count: int = 0
    # Configuration for agents (allows generic agent injection)
    initial_agent_config: Optional[Dict[str, Any]] = None
    review_agent_config: Optional[Dict[str, Any]] = None


def create_initial_agent_node(model, tools: List = None, prompt: str = None, name: str = "initial_agent"):
    """
    Creates an initial agent node using create_react_agent.
    This allows for generic React agent injection.
    
    Args:
        model: The language model to use
        tools: List of tools for the agent
        prompt: Custom prompt for the agent
        name: Name of the agent
    
    Returns:
        A React agent node function
    """
    if tools is None:
        tools = []
    
    if prompt is None:
        prompt = (
            "You are an initial agent responsible for generating responses to user queries. "
            "Provide comprehensive and helpful responses. "
            "Your output will be reviewed by another agent."
        )
    
    # Create the React agent
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name=name
    )
    
    def initial_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """Node function that wraps the React agent and updates iteration count."""
        # Increment iteration count
        new_iteration = state.get("iteration_count", 0) + 1
        
        # Invoke the React agent
        result = react_agent.invoke({"messages": state["messages"]})
        
        # Update state with agent response and iteration count
        return {
            "messages": result["messages"],
            "iteration_count": new_iteration,
            "review_status": "pending"
        }
    
    return initial_agent_node


def create_review_agent_node(model, tools: List = None, prompt: str = None, name: str = "review_agent"):
    """
    Creates a review agent node using create_react_agent.
    This agent evaluates the output from the initial agent.
    
    Args:
        model: The language model to use
        tools: List of tools for the agent (typically none needed for review)
        prompt: Custom prompt for the review agent
        name: Name of the agent
    
    Returns:
        A React agent node function for review
    """
    if tools is None:
        tools = []
    
    if prompt is None:
        prompt = (
            "You are a review agent responsible for evaluating responses from another agent. "
            "Analyze the conversation and the most recent response. "
            "Determine if the response is satisfactory and complete. "
            "Respond with either 'APPROVED' if the response is good, "
            "or 'NEEDS_REVISION' followed by specific feedback if improvements are needed. "
            "Be thorough but fair in your evaluation."
        )
    
    # Create the React agent for review
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name=name
    )
    
    def review_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """Node function that wraps the review React agent."""
        # Invoke the review agent
        result = react_agent.invoke({"messages": state["messages"]})
        
        # Extract the review decision from the agent's response
        if result["messages"]:
            review_content = result["messages"][-1].content.upper()
            if "APPROVED" in review_content:
                review_status = "approved"
            elif "NEEDS_REVISION" in review_content:
                review_status = "needs_revision"
            else:
                # Default to needs_revision if unclear
                review_status = "needs_revision"
        else:
            review_status = "needs_revision"
        
        return {
            "messages": result["messages"],
            "review_status": review_status
        }
    
    return review_agent_node


def route_based_on_review(state: ReviewWorkflowState) -> Literal["approved", "needs_revision"]:
    """
    Routing function for conditional edges based on review status.
    
    Args:
        state: Current workflow state
        
    Returns:
        "approved" if review passed, "needs_revision" if it needs to loop back
    """
    review_status = state.get("review_status", "needs_revision")
    
    if review_status == "approved":
        return "approved"
    else:
        return "needs_revision"


def create_two_stage_review_workflow(
    initial_model,
    review_model=None,
    initial_tools: List = None,
    review_tools: List = None,
    initial_prompt: str = None,
    review_prompt: str = None,
    max_iterations: int = 5
):
    """
    Creates a two-stage review workflow graph.
    
    Args:
        initial_model: Model for the initial agent
        review_model: Model for the review agent (defaults to initial_model)
        initial_tools: Tools for the initial agent
        review_tools: Tools for the review agent
        initial_prompt: Custom prompt for initial agent
        review_prompt: Custom prompt for review agent
        max_iterations: Maximum number of revision iterations
        
    Returns:
        Compiled LangGraph workflow
    """
    if review_model is None:
        review_model = initial_model
    
    # Create agent nodes
    initial_agent = create_initial_agent_node(
        model=initial_model,
        tools=initial_tools,
        prompt=initial_prompt,
        name="initial_agent"
    )
    
    review_agent = create_review_agent_node(
        model=review_model,
        tools=review_tools,
        prompt=review_prompt,
        name="review_agent"
    )
    
    # Create the StateGraph
    workflow = StateGraph(ReviewWorkflowState)
    
    # Add nodes
    workflow.add_node("initial_agent", initial_agent)
    workflow.add_node("review_agent", review_agent)
    
    # Add edges
    workflow.add_edge(START, "initial_agent")
    workflow.add_edge("initial_agent", "review_agent")
    
    # Add conditional edges based on review
    workflow.add_conditional_edges(
        "review_agent",
        route_based_on_review,
        {
            "approved": END,
            "needs_revision": "initial_agent"
        }
    )
    
    # Compile the graph
    return workflow.compile()


# Default configuration using Anthropic Claude
def create_default_workflow():
    """Creates a default two-stage review workflow with Claude."""
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    return create_two_stage_review_workflow(
        initial_model=model,
        review_model=model
    )


# Export the compiled graph as 'app' (required by LangGraph)
app = create_default_workflow()


# Example usage and configuration demonstrating generic design
if __name__ == "__main__":
    """
    Example usage showing how to inject arbitrary React agents
    by passing different models, tools, and prompts to the workflow.
    """
    
    # Example 1: Basic usage with default configuration
    print("=== Example 1: Basic Usage ===")
    basic_workflow = create_default_workflow()
    
    # Example 2: Custom prompts for different use cases
    print("=== Example 2: Custom Prompts ===")
    
    # Code review workflow
    code_review_workflow = create_two_stage_review_workflow(
        initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        initial_prompt=(
            "You are a senior software engineer. "
            "Review the code provided and suggest improvements. "
            "Focus on best practices, performance, and maintainability."
        ),
        review_prompt=(
            "You are a technical lead reviewing code suggestions. "
            "Evaluate if the code review is thorough and actionable. "
            "Respond with 'APPROVED' if the review is comprehensive, "
            "or 'NEEDS_REVISION' with specific areas to improve."
        )
    )
    
    # Content writing workflow
    content_writing_workflow = create_two_stage_review_workflow(
        initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        initial_prompt=(
            "You are a professional content writer. "
            "Create engaging, well-structured content based on the user's request. "
            "Focus on clarity, engagement, and value to the reader."
        ),
        review_prompt=(
            "You are an editor reviewing written content. "
            "Check for clarity, engagement, grammar, and structure. "
            "Respond with 'APPROVED' if the content meets publication standards, "
            "or 'NEEDS_REVISION' with specific feedback for improvement."
        )
    )
    
    # Example 3: Different models for different stages
    print("=== Example 3: Different Models ===")
    
    # Using different models (if available)
    mixed_model_workflow = create_two_stage_review_workflow(
        initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        review_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),  # Could be different model
        initial_prompt="You are a creative writer generating initial drafts.",
        review_prompt="You are a strict editor ensuring quality standards."
    )
    
    # Example 4: With tools (if you have tools available)
    print("=== Example 4: With Tools ===")
    
    # Example tools (you would import actual tools)
    # from langchain_community.tools import DuckDuckGoSearchRun
    # search_tool = DuckDuckGoSearchRun()
    
    # research_workflow = create_two_stage_review_workflow(
    #     initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    #     initial_tools=[search_tool],
    #     initial_prompt="You are a researcher. Use search tools to gather information.",
    #     review_prompt="You are a fact-checker reviewing research quality."
    # )
    
    print("All example workflows created successfully!")
    print("The 'app' variable contains the default compiled workflow.")
    print("Use app.invoke({'messages': [HumanMessage('Your query here')]}) to run it.")

"""
Two-Stage Agent Review Workflow

This module implements a generic two-stage review workflow using LangGraph:
1. First stage: React agent generates initial output
2. Second stage: Review agent evaluates the output quality
3. Conditional routing: Good output -> END, Bad output -> back to React agent

The workflow is designed to be generic and accept any React agent as a parameter.
"""

from typing import Literal, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent


def create_review_workflow(react_agent, max_iterations: int = 3):
    """
    Create a two-stage review workflow that accepts any React agent.
    
    Args:
        react_agent: A compiled LangGraph React agent
        max_iterations: Maximum number of review iterations before forcing completion
    
    Returns:
        Compiled LangGraph workflow with review loop
    """
    
    # Initialize the review model (using Anthropic as preferred)
    review_model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    # Extended state to track review iterations
    class ReviewState(MessagesState):
        iteration_count: int = 0
        original_query: str = ""
    
    def react_agent_node(state: ReviewState) -> Dict[str, Any]:
        """
        Execute the React agent with the current messages.
        """
        # Invoke the React agent with current state
        result = react_agent.invoke({"messages": state["messages"]})
        
        # Extract the messages from the result
        new_messages = result.get("messages", [])
        
        # Store original query on first iteration
        original_query = state.get("original_query", "")
        if not original_query and state["messages"]:
            # Find the first human message as the original query
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break
        
        return {
            "messages": new_messages,
            "original_query": original_query
        }
    
    def review_agent_node(state: ReviewState) -> Dict[str, Any]:
        """
        Review the React agent's output and determine if it's satisfactory.
        """
        # Get the last AI message (React agent's response)
        last_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_message = msg
                break
        
        if not last_message:
            # No AI response to review, mark as needing improvement
            review_message = AIMessage(
                content="No response found to review. The agent needs to provide an answer.",
                name="reviewer"
            )
            return {
                "messages": [review_message],
                "iteration_count": state.get("iteration_count", 0) + 1
            }
        
        # Create review prompt
        review_prompt = f"""
        Please review the following response to determine if it adequately answers the original query.
        
        Original Query: {state.get('original_query', 'Unknown')}
        
        Response to Review: {last_message.content}
        
        Evaluation Criteria:
        1. Does the response directly address the original query?
        2. Is the response complete and informative?
        3. Is the response accurate and well-reasoned?
        4. Are there any obvious errors or omissions?
        
        Respond with either:
        - "APPROVED: [brief reason]" if the response is satisfactory
        - "NEEDS_IMPROVEMENT: [specific feedback]" if the response needs work
        
        Be concise but specific in your evaluation.
        """
        
        # Get review from the review model
        review_response = review_model.invoke([HumanMessage(content=review_prompt)])
        
        # Create review message
        review_message = AIMessage(
            content=review_response.content,
            name="reviewer"
        )
        
        return {
            "messages": [review_message],
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    
    def should_continue(state: ReviewState) -> Literal["react_agent", "END"]:
        """
        Routing function to determine next step based on review.
        """
        # Check if we've exceeded max iterations
        if state.get("iteration_count", 0) >= max_iterations:
            return "END"
        
        # Get the last reviewer message
        last_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and getattr(msg, 'name', None) == 'reviewer':
                last_message = msg
                break
        
        if not last_message:
            # No review found, continue to react agent
            return "react_agent"
        
        # Check if the review approves the response
        review_content = last_message.content.upper()
        if review_content.startswith("APPROVED"):
            return "END"
        else:
            # Needs improvement, send back to react agent
            return "react_agent"
    
    def add_improvement_feedback(state: ReviewState) -> Dict[str, Any]:
        """
        Add feedback message to guide the React agent's improvement.
        """
        # Get the last reviewer message
        last_review = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and getattr(msg, 'name', None) == 'reviewer':
                last_review = msg
                break
        
        if last_review and last_review.content.upper().startswith("NEEDS_IMPROVEMENT"):
            # Extract feedback and create improvement message
            feedback = last_review.content
            improvement_message = HumanMessage(
                content=f"Please improve your previous response based on this feedback: {feedback}"
            )
            return {"messages": [improvement_message]}
        
        return {"messages": []}
    
    # Build the workflow graph
    workflow = StateGraph(ReviewState)
    
    # Add nodes
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_node("reviewer", review_agent_node)
    workflow.add_node("add_feedback", add_improvement_feedback)
    
    # Add edges
    workflow.add_edge(START, "react_agent")
    workflow.add_edge("react_agent", "reviewer")
    
    # Add conditional routing from reviewer
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "react_agent": "add_feedback",
            "END": END
        }
    )
    
    # Connect feedback back to react agent
    workflow.add_edge("add_feedback", "react_agent")
    
    return workflow.compile()


def create_dummy_react_agent():
    """
    Create a simple dummy React agent for testing purposes.
    """
    # Simple tools for the dummy agent
    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        return f"The weather in {location} is sunny and 72°F."
    
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            # Simple evaluation (in production, use a safer approach)
            result = eval(expression.replace("^", "**"))
            return f"The result of {expression} is {result}"
        except:
            return f"Could not calculate {expression}. Please provide a valid mathematical expression."
    
    # Create the dummy React agent
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    dummy_agent = create_react_agent(
        model=model,
        tools=[get_weather, calculate],
        prompt=(
            "You are a helpful assistant with access to weather and calculation tools. "
            "Use the available tools to answer user questions accurately and completely. "
            "Always provide clear, informative responses."
        )
    )
    
    return dummy_agent


# Create the default workflow with a dummy React agent
dummy_react_agent = create_dummy_react_agent()
app = create_review_workflow(dummy_react_agent)

# Export the compiled graph as 'app' (required for LangGraph deployment)
__all__ = ["app", "create_review_workflow", "create_dummy_react_agent"]

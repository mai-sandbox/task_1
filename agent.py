"""
Review-Based React Agent Workflow

This module implements a generic review-based workflow that can wrap any React agent
to add a review layer with feedback loops and retry logic.
"""

from typing import Literal, Dict, Any, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


# Extended state to track review workflow
class ReviewWorkflowState(MessagesState):
    """State for the review-based workflow with retry tracking."""
    retry_count: int = 0
    max_retries: int = 3
    current_output: str = ""
    review_feedback: str = ""
    workflow_complete: bool = False


class ReviewBasedAgent:
    """
    A generic review-based agent workflow that wraps any React agent
    to add review capabilities with feedback loops.
    """
    
    def __init__(
        self,
        initial_agent,
        review_agent=None,
        max_retries: int = 3,
        model_name: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize the review-based agent workflow.
        
        Args:
            initial_agent: The primary React agent that generates initial output
            review_agent: Optional custom review agent. If None, creates a default one
            max_retries: Maximum number of retry attempts
            model_name: Model to use for the default review agent
        """
        self.initial_agent = initial_agent
        self.max_retries = max_retries
        
        # Create default review agent if none provided
        if review_agent is None:
            model = ChatAnthropic(model=model_name, temperature=0)
            self.review_agent = create_react_agent(
                model=model,
                tools=[],  # Review agent doesn't need tools, just evaluation
                prompt=self._get_review_prompt(),
                name="review_agent"
            )
        else:
            self.review_agent = review_agent
        
        # Build the workflow graph
        self.graph = self._build_workflow()
    
    def _get_review_prompt(self) -> str:
        """Get the system prompt for the review agent."""
        return """You are a quality review agent. Your job is to evaluate the output from another agent and determine if it meets quality standards.

INSTRUCTIONS:
1. Carefully review the agent's response for:
   - Accuracy and correctness
   - Completeness of the answer
   - Clarity and coherence
   - Appropriate tone and helpfulness

2. Respond with EXACTLY one of these two words:
   - "continue" - if the output is satisfactory and meets quality standards
   - "retry" - if the output needs improvement and should be regenerated

3. After your decision, provide brief feedback explaining your reasoning.

Format your response as: [DECISION] - [BRIEF FEEDBACK]

Example: "continue - The response is accurate, complete, and well-structured."
Example: "retry - The response is too vague and doesn't fully address the user's question."
"""
    
    def _build_workflow(self) -> StateGraph:
        """Build the review-based workflow graph."""
        # Create the workflow graph
        workflow = StateGraph(ReviewWorkflowState)
        
        # Add nodes
        workflow.add_node("initial_agent", self._initial_agent_node)
        workflow.add_node("review_agent", self._review_agent_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.add_edge(START, "initial_agent")
        workflow.add_conditional_edges(
            "initial_agent",
            self._route_after_initial,
            {
                "review": "review_agent",
                "finalize": "finalize"
            }
        )
        workflow.add_conditional_edges(
            "review_agent",
            self._route_after_review,
            {
                "retry": "initial_agent",
                "continue": "finalize",
                "max_retries": "finalize"
            }
        )
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _initial_agent_node(self, state: ReviewWorkflowState) -> Dict[str, Any]:
        """Execute the initial agent and capture its output."""
        # Prepare messages for the initial agent
        agent_messages = state["messages"].copy()
        
        # If this is a retry, add feedback to help the agent improve
        if state["retry_count"] > 0 and state["review_feedback"]:
            feedback_msg = HumanMessage(
                content=f"Previous attempt needs improvement. Feedback: {state['review_feedback']}. Please provide a better response."
            )
            agent_messages.append(feedback_msg)
        
        # Run the initial agent
        result = self.initial_agent.invoke({"messages": agent_messages})
        
        # Extract the agent's response
        if result.get("messages"):
            latest_message = result["messages"][-1]
            if hasattr(latest_message, 'content'):
                current_output = latest_message.content
            else:
                current_output = str(latest_message)
        else:
            current_output = "No response generated"
        
        return {
            "messages": result.get("messages", state["messages"]),
            "current_output": current_output
        }
    
    def _review_agent_node(self, state: ReviewWorkflowState) -> Dict[str, Any]:
        """Execute the review agent to evaluate the output."""
        # Prepare review message
        review_prompt = f"""Please review this agent output:

USER QUESTION: {state['messages'][0].content if state['messages'] else 'No question provided'}

AGENT OUTPUT: {state['current_output']}

Evaluate the quality and provide your decision (continue/retry) with feedback."""
        
        review_messages = [HumanMessage(content=review_prompt)]
        
        # Run the review agent
        result = self.review_agent.invoke({"messages": review_messages})
        
        # Extract review feedback
        if result.get("messages"):
            review_response = result["messages"][-1]
            if hasattr(review_response, 'content'):
                review_feedback = review_response.content
            else:
                review_feedback = str(review_response)
        else:
            review_feedback = "retry - No review response generated"
        
        return {
            "review_feedback": review_feedback,
            "retry_count": state["retry_count"] + 1
        }
    
    def _finalize_node(self, state: ReviewWorkflowState) -> Dict[str, Any]:
        """Finalize the workflow and prepare the final output."""
        return {
            "workflow_complete": True
        }
    
    def _route_after_initial(self, state: ReviewWorkflowState) -> str:
        """Route after initial agent execution."""
        # If we've reached max retries, finalize
        if state["retry_count"] >= self.max_retries:
            return "finalize"
        
        # Otherwise, send to review
        return "review"
    
    def _route_after_review(self, state: ReviewWorkflowState) -> str:
        """Route after review agent evaluation."""
        # Check if we've hit max retries
        if state["retry_count"] >= self.max_retries:
            return "max_retries"
        
        # Parse the review decision
        review_feedback = state.get("review_feedback", "").lower()
        
        if "continue" in review_feedback:
            return "continue"
        elif "retry" in review_feedback:
            return "retry"
        else:
            # Default to continue if unclear
            return "continue"


# Example usage and default setup
def create_dummy_agent():
    """Create a dummy React agent for demonstration purposes."""
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
    
    def dummy_tool(query: str) -> str:
        """A dummy tool that provides basic responses."""
        return f"Dummy response for: {query}"
    
    return create_react_agent(
        model=model,
        tools=[dummy_tool],
        prompt="You are a helpful assistant. Provide clear and concise answers to user questions.",
        name="dummy_agent"
    )


# Create the default review-based workflow
def create_review_workflow(
    initial_agent=None,
    review_agent=None,
    max_retries: int = 3
):
    """
    Create a review-based workflow with the specified agents.
    
    Args:
        initial_agent: The primary agent. If None, creates a dummy agent
        review_agent: The review agent. If None, creates a default review agent
        max_retries: Maximum retry attempts
    
    Returns:
        Compiled LangGraph workflow
    """
    if initial_agent is None:
        initial_agent = create_dummy_agent()
    
    review_workflow = ReviewBasedAgent(
        initial_agent=initial_agent,
        review_agent=review_agent,
        max_retries=max_retries
    )
    
    return review_workflow.graph


# Export the compiled graph as 'app' for deployment
app = create_review_workflow()


if __name__ == "__main__":
    # Example usage
    print("Review-Based React Agent Workflow created successfully!")
    print("The workflow includes:")
    print("1. Initial React agent for generating responses")
    print("2. Review agent for quality evaluation")
    print("3. Conditional routing with retry logic")
    print("4. Maximum retry limit to prevent infinite loops")
    print("\nGraph exported as 'app' variable for deployment.")

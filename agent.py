"""
Two-Stage Agent Review Workflow Implementation

This module implements a generic two-stage review workflow using LangGraph where:
1. An initial React agent processes the user input
2. A review agent evaluates the output
3. If approved, the workflow ends; if rejected, it loops back with feedback
"""

from typing import Annotated, Literal, Dict, Any, List
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


class ReviewState(MessagesState):
    """
    Extended state for the two-stage review workflow.
    
    Inherits from MessagesState to get the messages field with add_messages reducer.
    Adds additional fields for tracking the review process.
    """
    review_status: str  # "pending", "approved", "rejected"
    attempt_count: int  # Number of attempts made
    feedback: str  # Feedback from the review agent


class TwoStageReviewWorkflow:
    """
    A generic two-stage review workflow that accepts any React agent.
    
    The workflow consists of:
    1. Initial agent processes the user input
    2. Review agent evaluates the output
    3. Conditional routing based on review decision
    """
    
    def __init__(self, initial_agent, max_attempts: int = 3):
        """
        Initialize the two-stage review workflow.
        
        Args:
            initial_agent: A compiled LangGraph agent (from create_react_agent)
            max_attempts: Maximum number of retry attempts (default: 3)
        """
        self.initial_agent = initial_agent
        self.max_attempts = max_attempts
        
        # Create the review agent with a specific prompt for evaluation
        self.review_agent = create_react_agent(
            model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
            tools=[],  # Review agent doesn't need tools, just evaluation
            prompt=self._get_review_prompt()
        )
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _get_review_prompt(self) -> str:
        """Get the system prompt for the review agent."""
        return """You are a review agent responsible for evaluating the quality and accuracy of responses from another agent.

Your task is to:
1. Carefully review the agent's response to the user's question
2. Determine if the response is satisfactory, accurate, and helpful
3. Respond with EXACTLY one of these formats:

For APPROVAL:
APPROVED: The response is satisfactory.

For REJECTION:
REJECTED: [Specific feedback about what needs to be improved]

Be thorough but concise in your evaluation. Focus on accuracy, helpfulness, and completeness."""

    def _build_graph(self) -> StateGraph:
        """Build the StateGraph for the two-stage review workflow."""
        
        # Create the state graph
        workflow = StateGraph(ReviewState)
        
        # Add nodes
        workflow.add_node("initial_agent", self._initial_agent_node)
        workflow.add_node("review_agent", self._review_agent_node)
        workflow.add_node("prepare_retry", self._prepare_retry_node)
        
        # Set entry point
        workflow.add_edge(START, "initial_agent")
        
        # Add conditional edges for review decision
        workflow.add_conditional_edges(
            "review_agent",
            self._review_decision,
            {
                "approved": END,
                "rejected": "prepare_retry",
                "max_attempts": END
            }
        )
        
        # Edge from prepare_retry back to initial_agent
        workflow.add_edge("prepare_retry", "initial_agent")
        
        # Edge from initial_agent to review_agent
        workflow.add_edge("initial_agent", "review_agent")
        
        return workflow.compile()
    
    def _initial_agent_node(self, state: ReviewState) -> Dict[str, Any]:
        """
        Node that runs the initial agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            State update with the agent's response
        """
        # Prepare messages for the initial agent
        # If this is a retry, include the feedback
        messages = state["messages"].copy()
        
        if state.get("feedback") and state.get("attempt_count", 0) > 0:
            # Add feedback as a system message for retry attempts
            feedback_msg = SystemMessage(
                content=f"Previous attempt was rejected. Feedback: {state['feedback']}. Please improve your response based on this feedback."
            )
            messages.append(feedback_msg)
        
        # Invoke the initial agent
        result = self.initial_agent.invoke({"messages": messages})
        
        # Extract the agent's response
        agent_response = result["messages"][-1]
        
        return {
            "messages": [agent_response],
            "review_status": "pending"
        }
    
    def _review_agent_node(self, state: ReviewState) -> Dict[str, Any]:
        """
        Node that runs the review agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            State update with review decision and feedback
        """
        # Get the original user message and the agent's response
        original_messages = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
        user_message = original_messages[0]  # Original user input
        agent_response = original_messages[-1]  # Latest agent response
        
        # Create review context
        review_messages = [
            HumanMessage(content=f"""Please review this interaction:

User Question: {user_message.content}

Agent Response: {agent_response.content}

Please evaluate if this response adequately answers the user's question.""")
        ]
        
        # Get review decision
        review_result = self.review_agent.invoke({"messages": review_messages})
        review_response = review_result["messages"][-1].content
        
        # Parse the review response
        if review_response.startswith("APPROVED"):
            review_status = "approved"
            feedback = ""
        elif review_response.startswith("REJECTED"):
            review_status = "rejected"
            # Extract feedback after "REJECTED: "
            feedback = review_response.replace("REJECTED: ", "").strip()
        else:
            # Fallback parsing
            if "approved" in review_response.lower():
                review_status = "approved"
                feedback = ""
            else:
                review_status = "rejected"
                feedback = review_response
        
        return {
            "review_status": review_status,
            "feedback": feedback,
            "messages": [AIMessage(content=f"Review: {review_response}")]
        }
    
    def _prepare_retry_node(self, state: ReviewState) -> Dict[str, Any]:
        """
        Node that prepares for a retry attempt.
        
        Args:
            state: Current workflow state
            
        Returns:
            State update with incremented attempt count
        """
        current_attempts = state.get("attempt_count", 0)
        return {
            "attempt_count": current_attempts + 1
        }
    
    def _review_decision(self, state: ReviewState) -> Literal["approved", "rejected", "max_attempts"]:
        """
        Conditional edge function that determines the next step based on review.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node to execute
        """
        # Check if max attempts reached
        if state.get("attempt_count", 0) >= self.max_attempts:
            return "max_attempts"
        
        # Route based on review status
        if state.get("review_status") == "approved":
            return "approved"
        else:
            return "rejected"


def create_default_initial_agent():
    """
    Create a default initial agent for demonstration purposes.
    This is a simple React agent that can answer general questions.
    """
    return create_react_agent(
        model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        tools=[],  # No tools for this simple example
        prompt="You are a helpful assistant. Answer questions clearly and concisely."
    )


# Create the default workflow instance
default_initial_agent = create_default_initial_agent()
default_workflow = TwoStageReviewWorkflow(default_initial_agent)

# Export the compiled graph as 'app' as required by LangGraph conventions
app = default_workflow.graph

"""
Two-Stage Agent Review Workflow

This module implements a generic two-stage review workflow that:
1. Accepts any arbitrary React agent as input parameter
2. Runs the React agent to get initial output
3. Uses a review agent to evaluate the output quality
4. Implements conditional logic to either finish (if good) or retry with feedback
5. Uses MessagesState for state management and conditional edges for routing decisions
6. Exports the compiled graph as 'app' variable

The workflow is designed to be deployment-ready and follows LangGraph best practices.
"""

from typing import Literal, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


class ReviewWorkflowState(MessagesState):
    """
    State schema for the two-stage review workflow.
    Extends MessagesState to include additional fields for workflow control.
    """
    # Track the current iteration count to prevent infinite loops
    iteration_count: int
    # Track whether we're in initial run or retry
    is_retry: bool
    # Store feedback from review agent for retry attempts
    review_feedback: str
    # Maximum number of retry attempts allowed
    max_iterations: int


def create_two_stage_review_workflow(
    react_agent: Optional[Any] = None,
    model_name: str = "claude-3-5-sonnet-20241022",
    max_iterations: int = 3
):
    """
    Creates a two-stage review workflow that can work with any React agent.
    
    Args:
        react_agent: Optional React agent to use. If None, creates a default one.
        model_name: Model to use for the review agent
        max_iterations: Maximum number of retry attempts
    
    Returns:
        Compiled LangGraph workflow
    """
    
    # Initialize the model for review agent
    model = ChatAnthropic(model=model_name)
    
    # Create a default React agent if none provided (for demonstration)
    if react_agent is None:
        react_agent = create_react_agent(
            model=model,
            tools=[],  # No tools for the default demo agent
            prompt="You are a helpful assistant. Answer the user's question to the best of your ability."
        )
    
    def initial_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """
        Node that runs the initial React agent or handles retry with feedback.
        """
        messages = state["messages"].copy()
        
        # If this is a retry, add the feedback as context
        if state.get("is_retry", False) and state.get("review_feedback"):
            feedback_message = SystemMessage(
                content=f"Previous attempt needs improvement. Feedback: {state['review_feedback']}. "
                        f"Please revise your response based on this feedback."
            )
            messages.append(feedback_message)
        
        # Run the React agent
        result = react_agent.invoke({"messages": messages})
        
        # Extract the final response - get all new messages from the agent
        agent_messages = result.get("messages", [])
        if agent_messages:
            # Find new messages that weren't in the original input
            original_count = len(messages)
            new_agent_messages = agent_messages[original_count:] if len(agent_messages) > original_count else [agent_messages[-1]]
        else:
            # Fallback if no messages returned
            new_agent_messages = [AIMessage(content="No response generated.")]
        
        return {
            "messages": state["messages"] + new_agent_messages,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "is_retry": False  # Reset retry flag after processing
        }
    
    def review_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """
        Node that reviews the output from the initial agent and provides feedback.
        """
        # Get the last AI message (the response to review)
        messages = state["messages"]
        last_ai_message = None
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            # No AI message to review, mark as needing retry
            return {
                "review_feedback": "No response found to review. Please provide a proper response.",
                "is_retry": True
            }
        
        # Create review prompt
        review_prompt = f"""
        Please review the following response for quality, accuracy, and completeness:
        
        Response to review: {last_ai_message.content}
        
        Evaluate the response based on:
        1. Does it directly address the user's question/request?
        2. Is the information accurate and helpful?
        3. Is the response complete and well-structured?
        4. Is the tone appropriate?
        
        Respond with either:
        - "APPROVED" if the response is good quality and ready to be delivered
        - "NEEDS_IMPROVEMENT: [specific feedback]" if the response needs to be revised
        
        Be specific about what needs improvement if you reject it.
        """
        
        review_messages = [
            SystemMessage(content="You are a quality review agent. Your job is to evaluate responses and provide constructive feedback."),
            HumanMessage(content=review_prompt)
        ]
        
        # Get review from the model
        review_result = model.invoke(review_messages)
        review_content = review_result.content
        
        # Parse the review result
        if review_content.startswith("APPROVED"):
            return {
                "review_feedback": "Response approved",
                "is_retry": False
            }
        elif "NEEDS_IMPROVEMENT:" in review_content:
            feedback = review_content.split("NEEDS_IMPROVEMENT:", 1)[1].strip()
            return {
                "review_feedback": feedback,
                "is_retry": True
            }
        else:
            # Default to needing improvement if unclear
            return {
                "review_feedback": f"Review unclear. Original review: {review_content}",
                "is_retry": True
            }
    
    def should_continue(state: ReviewWorkflowState) -> Literal["continue", "finish", "max_iterations_reached"]:
        """
        Conditional edge function that determines the next step based on review results.
        """
        # Check if we've reached max iterations
        if state.get("iteration_count", 0) >= state.get("max_iterations", max_iterations):
            return "max_iterations_reached"
        
        # Check if review approved the response
        if not state.get("is_retry", False):
            return "finish"
        
        # Otherwise, continue with retry
        return "continue"
    
    def finalize_response_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """
        Node that finalizes the response when approved or max iterations reached.
        """
        messages = state["messages"]
        
        # If we reached max iterations, add a note
        if state.get("iteration_count", 0) >= state.get("max_iterations", max_iterations):
            final_message = AIMessage(
                content=f"Response after {state.get('iteration_count', 0)} iterations. "
                        f"Note: Maximum iteration limit reached."
            )
            messages = messages + [final_message]
        
        return {"messages": messages}
    
    # Build the workflow graph
    workflow = StateGraph(ReviewWorkflowState)
    
    # Add nodes
    workflow.add_node("initial_agent", initial_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    workflow.add_node("finalize_response", finalize_response_node)
    
    # Add edges
    workflow.add_edge(START, "initial_agent")
    workflow.add_edge("initial_agent", "review_agent")
    
    # Add conditional edges for the review decision
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "continue": "initial_agent",  # Retry with feedback
            "finish": "finalize_response",  # Approved, finalize
            "max_iterations_reached": "finalize_response"  # Max iterations, finalize anyway
        }
    )
    
    workflow.add_edge("finalize_response", END)
    
    return workflow.compile()


# Create the default workflow instance and export as 'app'
app = create_two_stage_review_workflow()


# Alternative function to create workflow with custom React agent
def create_workflow_with_custom_agent(react_agent, max_iterations: int = 3):
    """
    Convenience function to create workflow with a custom React agent.
    
    Args:
        react_agent: The React agent to use in the workflow
        max_iterations: Maximum number of retry attempts
    
    Returns:
        Compiled LangGraph workflow
    """
    return create_two_stage_review_workflow(
        react_agent=react_agent,
        max_iterations=max_iterations
    )


# Example usage function (for testing)
def example_usage():
    """
    Example of how to use the two-stage review workflow.
    """
    # Example 1: Using the default workflow
    initial_state = {
        "messages": [HumanMessage("What is the capital of France?")],
        "iteration_count": 0,
        "original_input": "What is the capital of France?",
        "max_iterations": 3
    }
    
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Test the workflow
    print("Testing two-stage review workflow...")
    result = example_usage()
    print("Final result:", result)




"""
Two-Stage Agent Review Workflow

This module implements a generic two-stage workflow where:
1. A ReAct agent generates initial output
2. A review agent evaluates the output quality
3. If good, the workflow finishes; if not, it loops back to the original agent

The workflow is designed to be generic and accept any ReAct agent.
"""

from typing import TypedDict, List, Literal, Optional, Callable, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


class ReviewWorkflowState(TypedDict):
    """State schema for the two-stage review workflow."""
    messages: List[BaseMessage]
    review_status: Optional[Literal["pending", "approved", "needs_revision"]]
    iteration_count: int
    max_iterations: int
    original_query: str
    current_output: Optional[str]


def create_review_workflow(
    react_agent: Optional[Callable] = None,
    review_model: Optional[ChatOpenAI] = None,
    max_iterations: int = 3
) -> StateGraph:
    """
    Create a two-stage review workflow graph.
    
    Args:
        react_agent: The ReAct agent to use for initial output generation.
                    If None, a default agent will be created.
        review_model: The model to use for reviewing outputs.
                     If None, a default ChatOpenAI model will be used.
        max_iterations: Maximum number of iterations before stopping.
    
    Returns:
        Compiled StateGraph representing the workflow.
    """
    
    # Default review model
    if review_model is None:
        review_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Default ReAct agent with basic tools
    if react_agent is None:
        @tool
        def search_tool(query: str) -> str:
            """A simple search tool that returns a mock response."""
            return f"Search results for '{query}': This is a mock search result."
        
        @tool
        def calculator_tool(expression: str) -> str:
            """A simple calculator tool."""
            try:
                result = eval(expression)
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error calculating {expression}: {str(e)}"
        
        tools = [search_tool, calculator_tool]
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        react_agent = create_react_agent(model, tools)
    
    def react_agent_node(state: ReviewWorkflowState) -> ReviewWorkflowState:
        """Execute the ReAct agent to generate or refine output."""
        messages = state["messages"]
        
        # If this is a revision iteration, add context about previous feedback
        if state["review_status"] == "needs_revision" and state["iteration_count"] > 0:
            revision_prompt = SystemMessage(
                content="The previous response needs revision based on the review feedback. "
                       "Please improve your response addressing the concerns mentioned."
            )
            messages = [revision_prompt] + messages
        
        # Execute the ReAct agent
        result = react_agent.invoke({"messages": messages})
        
        # Extract the final response
        if result and "messages" in result:
            agent_messages = result["messages"]
            if agent_messages:
                current_output = agent_messages[-1].content
            else:
                current_output = "No output generated"
        else:
            current_output = str(result)
        
        return {
            **state,
            "messages": messages + [AIMessage(content=current_output)],
            "current_output": current_output,
            "review_status": "pending"
        }
    
    def review_agent_node(state: ReviewWorkflowState) -> ReviewWorkflowState:
        """Review the output from the ReAct agent."""
        current_output = state["current_output"]
        original_query = state["original_query"]
        
        review_prompt = f"""
        Please review the following response to determine if it adequately addresses the user's query.
        
        Original Query: {original_query}
        
        Response to Review: {current_output}
        
        Evaluation Criteria:
        1. Does the response directly address the user's question?
        2. Is the information accurate and helpful?
        3. Is the response complete and well-structured?
        4. Are there any obvious errors or omissions?
        
        Respond with either:
        - "APPROVED" if the response is satisfactory
        - "NEEDS_REVISION: [specific feedback]" if the response needs improvement
        
        Be specific about what needs to be improved if revision is needed.
        """
        
        review_result = review_model.invoke([HumanMessage(content=review_prompt)])
        review_content = review_result.content.strip()
        
        # Determine review status
        if review_content.startswith("APPROVED"):
            review_status = "approved"
        else:
            review_status = "needs_revision"
        
        # Add review message to conversation
        review_message = AIMessage(
            content=f"Review: {review_content}",
            name="reviewer"
        )
        
        return {
            **state,
            "messages": state["messages"] + [review_message],
            "review_status": review_status
        }
    
    def decision_node(state: ReviewWorkflowState) -> ReviewWorkflowState:
        """Make routing decisions based on review status and iteration count."""
        return {
            **state,
            "iteration_count": state["iteration_count"] + 1
        }
    
    def should_continue(state: ReviewWorkflowState) -> str:
        """Determine the next step based on review status and iteration count."""
        review_status = state["review_status"]
        iteration_count = state["iteration_count"]
        max_iterations = state["max_iterations"]
        
        # If approved, end the workflow
        if review_status == "approved":
            return END
        
        # If max iterations reached, end the workflow
        if iteration_count >= max_iterations:
            return END
        
        # If needs revision and under max iterations, go back to react agent
        if review_status == "needs_revision":
            return "react_agent"
        
        # Default case - should not happen
        return END
    
    # Create the workflow graph
    workflow = StateGraph(ReviewWorkflowState)
    
    # Add nodes
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    workflow.add_node("decision", decision_node)
    
    # Set entry point
    workflow.set_entry_point("react_agent")
    
    # Add edges
    workflow.add_edge("react_agent", "review_agent")
    workflow.add_edge("review_agent", "decision")
    
    # Add conditional edge from decision node
    workflow.add_conditional_edges(
        "decision",
        should_continue,
        {
            "react_agent": "react_agent",
            END: END
        }
    )
    
    return workflow


def create_default_workflow() -> StateGraph:
    """Create a default workflow with standard settings."""
    return create_review_workflow()


# Create and compile the default workflow
workflow = create_default_workflow()
app = workflow.compile()


def invoke_workflow(query: str, max_iterations: int = 3) -> dict:
    """
    Convenience function to invoke the workflow with a query.
    
    Args:
        query: The user's query/request
        max_iterations: Maximum number of revision iterations
    
    Returns:
        The final state of the workflow
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "review_status": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "original_query": query,
        "current_output": None
    }
    
    return app.invoke(initial_state)


if __name__ == "__main__":
    # Example usage
    result = invoke_workflow("What is the capital of France and what is 2+2?")
    print("Final result:")
    print(f"Status: {result['review_status']}")
    print(f"Iterations: {result['iteration_count']}")
    print(f"Final output: {result['current_output']}")

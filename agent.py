"""
LangGraph agent that implements a React agent with reviewer pattern.
The workflow is: React agent -> Reviewer -> Decision (finish or retry)
"""

from typing import TypedDict, List, Literal, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


class State(TypedDict):
    """State schema for the dual-agent system."""
    messages: Annotated[List[BaseMessage], add_messages]
    react_output: Optional[str]
    review_feedback: Optional[str]
    iteration_count: int
    max_iterations: int
    is_approved: bool
    react_agent_graph: Optional[object]


def create_dual_agent_system(react_agent, max_iterations: int = 3):
    """
    Creates a dual-agent system where a React agent generates output
    and a reviewer agent evaluates it.
    
    Args:
        react_agent: A compiled LangGraph agent (should be a React-style agent)
        max_iterations: Maximum number of retry iterations
    
    Returns:
        Compiled LangGraph application
    """
    
    # Initialize the LLM for the reviewer
    reviewer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def run_react_agent(state: State):
        """Run the React agent on the current messages."""
        try:
            # Use the provided react agent
            react_result = react_agent.invoke({
                "messages": state["messages"]
            })
            
            # Extract the final response
            if react_result and "messages" in react_result:
                last_message = react_result["messages"][-1]
                if hasattr(last_message, 'content'):
                    react_output = last_message.content
                else:
                    react_output = str(last_message)
            else:
                react_output = str(react_result)
            
            return {
                "react_output": react_output,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
        except Exception as e:
            return {
                "react_output": f"Error in React agent: {str(e)}",
                "iteration_count": state.get("iteration_count", 0) + 1
            }
    
    def review_output(state: State):
        """Review the React agent's output."""
        original_query = state["messages"][0].content if state["messages"] else "Unknown query"
        react_output = state["react_output"]
        
        review_prompt = f"""
        You are a quality reviewer. Evaluate the following response to determine if it adequately addresses the user's query.
        
        Original Query: {original_query}
        
        Response to Review: {react_output}
        
        Evaluation Criteria:
        1. Does the response directly answer the user's question?
        2. Is the response accurate and well-reasoned?
        3. Is the response complete and not missing important information?
        4. Is the response clear and understandable?
        
        Respond with either:
        - "APPROVED: [brief explanation]" if the response is satisfactory
        - "REJECTED: [specific feedback on what needs improvement]" if it needs work
        
        Be constructive and specific in your feedback.
        """
        
        try:
            review_result = reviewer_llm.invoke([HumanMessage(content=review_prompt)])
            review_feedback = review_result.content
            
            is_approved = review_feedback.upper().startswith("APPROVED")
            
            return {
                "review_feedback": review_feedback,
                "is_approved": is_approved
            }
        except Exception as e:
            return {
                "review_feedback": f"Error in review: {str(e)}",
                "is_approved": False
            }
    
    def prepare_retry(state: State):
        """Prepare the system for a retry with reviewer feedback."""
        original_message = state["messages"][0]
        review_feedback = state["review_feedback"]
        
        retry_message = HumanMessage(
            content=f"{original_message.content}\n\n"
                   f"Previous attempt feedback: {review_feedback}\n"
                   f"Please improve your response based on this feedback."
        )
        
        return {
            "messages": [retry_message],
            "react_output": None,
            "review_feedback": None
        }
    
    def finalize_response(state: State):
        """Finalize the approved response."""
        final_message = AIMessage(content=state["react_output"])
        return {
            "messages": [final_message]
        }
    
    def should_retry(state: State) -> Literal["retry", "finish"]:
        """Determine if we should retry or finish."""
        if state["is_approved"]:
            return "finish"
        elif state["iteration_count"] >= state.get("max_iterations", max_iterations):
            return "finish"
        else:
            return "retry"
    
    # Build the workflow graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("react_agent", run_react_agent)
    workflow.add_node("reviewer", review_output)
    workflow.add_node("prepare_retry", prepare_retry)
    workflow.add_node("finalize", finalize_response)
    
    # Add edges
    workflow.set_entry_point("react_agent")
    workflow.add_edge("react_agent", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        should_retry,
        {
            "retry": "prepare_retry",
            "finish": "finalize"
        }
    )
    workflow.add_edge("prepare_retry", "react_agent")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# Default implementation using a simple React agent
def create_default_react_agent():
    """Create a default React agent for demonstration."""
    from langchain_core.tools import tool
    
    @tool
    def search_tool(query: str) -> str:
        """Search for information about a topic."""
        return f"Search results for '{query}': This is a simulated search result."
    
    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except:
            return f"Could not calculate: {expression}"
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_tool, calculator_tool]
    
    return create_react_agent(llm, tools)


# Create the main application
def create_app(react_agent=None, max_iterations: int = 3):
    """
    Create the dual-agent application.
    
    Args:
        react_agent: Optional custom React agent. If None, uses default.
        max_iterations: Maximum retry iterations.
    
    Returns:
        Compiled LangGraph application
    """
    if react_agent is None:
        react_agent = create_default_react_agent()
    
    return create_dual_agent_system(react_agent, max_iterations)


# Export the default app
app = create_app()
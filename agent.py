from typing import TypedDict, List, Annotated, Literal, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import operator
import os
from dotenv import load_dotenv

load_dotenv()


class ReviewState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    initial_response: Optional[str]
    review_feedback: Optional[str] 
    iteration_count: int
    max_iterations: int
    is_approved: bool


@tool
def dummy_tool(query: str) -> str:
    """A dummy tool for demonstration purposes."""
    return f"Processed: {query}"


def create_react_agent_node(llm: Optional[ChatOpenAI] = None, tools: Optional[List] = None):
    """Create a generic ReAct agent that can be customized with different tools."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if tools is None:
        tools = [dummy_tool]
    
    react_agent = create_react_agent(llm, tools)
    
    def react_node(state: ReviewState):
        messages = state["messages"]
        
        if state.get("review_feedback"):
            feedback_message = SystemMessage(
                content=f"Previous response was not approved. Reviewer feedback: {state['review_feedback']}. Please improve your response."
            )
            messages = messages + [feedback_message]
        
        result = react_agent.invoke({"messages": messages})
        
        response_content = result["messages"][-1].content if result["messages"] else ""
        
        return {
            "messages": result["messages"],
            "initial_response": response_content,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    
    return react_node


def create_reviewer_node(llm: Optional[ChatOpenAI] = None):
    """Create a reviewer agent that evaluates the ReAct agent's output."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def reviewer_node(state: ReviewState):
        initial_response = state.get("initial_response", "")
        original_query = state["messages"][0].content if state["messages"] else ""
        
        review_prompt = f"""
        You are a quality reviewer. Evaluate the following response to determine if it adequately addresses the user's request.
        
        Original Request: {original_query}
        
        Response to Review: {initial_response}
        
        Evaluate if the response:
        1. Directly addresses the user's request
        2. Is complete and accurate
        3. Is clear and well-structured
        
        Respond with ONLY:
        - "APPROVED" if the response is satisfactory
        - "NEEDS_REVISION: [specific feedback]" if improvements are needed
        
        Be reasonable - don't be overly critical for minor issues.
        """
        
        review_result = llm.invoke([SystemMessage(content=review_prompt)])
        review_content = review_result.content.strip()
        
        is_approved = review_content.startswith("APPROVED")
        feedback = None if is_approved else review_content.replace("NEEDS_REVISION:", "").strip()
        
        return {
            "messages": [AIMessage(content=f"Review: {review_content}")],
            "is_approved": is_approved,
            "review_feedback": feedback
        }
    
    return reviewer_node


def should_continue(state: ReviewState) -> Literal["reviewer", "end"]:
    """Determine if we should continue to review or end."""
    if state.get("iteration_count", 0) == 0:
        return "reviewer"
    
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return "end"
    
    if state.get("is_approved", False):
        return "end"
    
    return "reviewer"


def should_retry(state: ReviewState) -> Literal["react_agent", "end"]:
    """Determine if we should retry with the ReAct agent or end."""
    if state.get("is_approved", False):
        return "end"
    
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return "end"
    
    return "react_agent"


def build_review_loop_graph(
    react_agent_llm: Optional[ChatOpenAI] = None,
    reviewer_llm: Optional[ChatOpenAI] = None,
    tools: Optional[List] = None,
    max_iterations: int = 3
):
    """Build the main review loop graph."""
    workflow = StateGraph(ReviewState)
    
    react_node = create_react_agent_node(react_agent_llm, tools)
    reviewer_node = create_reviewer_node(reviewer_llm)
    
    def initialize_state(state: ReviewState):
        """Initialize the state with default values."""
        return {
            "max_iterations": max_iterations,
            "iteration_count": 0,
            "is_approved": False
        }
    
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("react_agent", react_node)
    workflow.add_node("reviewer", reviewer_node)
    
    workflow.set_entry_point("initialize")
    
    workflow.add_edge("initialize", "react_agent")
    
    workflow.add_conditional_edges(
        "react_agent",
        should_continue,
        {
            "reviewer": "reviewer",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "reviewer",
        should_retry,
        {
            "react_agent": "react_agent",
            "end": END
        }
    )
    
    return workflow.compile()


if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment variables.")
    print("Please set it in your .env file or environment.")
    print("Example: cp .env.example .env and add your API key")

app = build_review_loop_graph()
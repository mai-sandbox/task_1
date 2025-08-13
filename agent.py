from typing import TypedDict, Annotated, Sequence, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
import operator
import os


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    react_output: str
    review_feedback: str
    iteration_count: int
    max_iterations: int
    llm: Optional[BaseChatModel]
    tools: Optional[list]


@tool
def example_tool(query: str) -> str:
    """A simple example tool that can be replaced with any tools needed."""
    return f"Processing query: {query}"


def react_agent_node(state: AgentState) -> dict:
    """Run the React agent to generate initial output."""
    messages = state["messages"]
    
    # Use provided LLM and tools, or defaults
    llm = state.get("llm")
    tools = state.get("tools", [example_tool])
    
    # Create default LLM if not provided
    if llm is None:
        # Try to get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        else:
            # Fallback to assuming it's configured in environment
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create the React agent
    react_agent = create_react_agent(llm, tools)
    
    # Get the last human message
    human_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    
    if human_msg:
        # Invoke the React agent
        result = react_agent.invoke({"messages": [human_msg]})
        
        # Extract the output
        if result["messages"]:
            last_message = result["messages"][-1]
            output = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            output = "No output generated"
    else:
        output = "No human message found"
    
    return {
        "react_output": output,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def review_agent_node(state: AgentState) -> dict:
    """Review the React agent's output and provide feedback."""
    react_output = state["react_output"]
    messages = state["messages"]
    
    # Use provided LLM or create default
    llm = state.get("llm")
    if llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Construct review prompt
    review_prompt = f"""
    Please review the following output from a React agent:
    
    Original Request: {next((msg.content for msg in messages if isinstance(msg, HumanMessage)), 'N/A')}
    
    React Agent Output:
    {react_output}
    
    Evaluate whether this output:
    1. Adequately addresses the original request
    2. Is accurate and complete
    3. Is clear and well-structured
    
    If the output is satisfactory, respond with "APPROVED".
    If the output needs improvement, respond with "NEEDS_IMPROVEMENT: [specific feedback on what to improve]".
    """
    
    review_result = llm.invoke(review_prompt)
    feedback = review_result.content
    
    return {"review_feedback": feedback}


def should_continue(state: AgentState) -> Literal["react", "end"]:
    """Determine whether to continue with another iteration or end."""
    feedback = state.get("review_feedback", "")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # Check if we've reached max iterations
    if iteration_count >= max_iterations:
        return "end"
    
    # Check if the output was approved
    if "APPROVED" in feedback:
        return "end"
    
    # Otherwise, continue with improvements
    return "react"


def feedback_processor_node(state: AgentState) -> dict:
    """Process feedback and prepare for the next iteration."""
    feedback = state.get("review_feedback", "")
    react_output = state.get("react_output", "")
    messages = state["messages"]
    
    # Extract improvement suggestions from feedback
    if "NEEDS_IMPROVEMENT:" in feedback:
        improvement_feedback = feedback.split("NEEDS_IMPROVEMENT:")[1].strip()
    else:
        improvement_feedback = feedback
    
    # Create a new message with the feedback for the React agent
    feedback_message = HumanMessage(
        content=f"""
        Previous output: {react_output}
        
        Feedback from reviewer: {improvement_feedback}
        
        Please improve your response based on this feedback.
        """
    )
    
    # Add the feedback message to the state
    new_messages = list(messages) + [feedback_message]
    
    return {"messages": new_messages}


def final_output_node(state: AgentState) -> dict:
    """Prepare the final output."""
    react_output = state.get("react_output", "")
    review_feedback = state.get("review_feedback", "")
    
    final_message = AIMessage(
        content=f"""
        Final Output:
        {react_output}
        
        Review Status: {'Approved' if 'APPROVED' in review_feedback else 'Max iterations reached'}
        """
    )
    
    return {"messages": [final_message]}


# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("react", react_agent_node)
graph.add_node("review", review_agent_node)
graph.add_node("feedback_processor", feedback_processor_node)
graph.add_node("final_output", final_output_node)

# Set entry point
graph.set_entry_point("react")

# Add edges
graph.add_edge("react", "review")
graph.add_conditional_edges(
    "review",
    should_continue,
    {
        "react": "feedback_processor",
        "end": "final_output"
    }
)
graph.add_edge("feedback_processor", "react")
graph.add_edge("final_output", END)

# Compile the graph
app = graph.compile()
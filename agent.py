"""
Two-Stage Agent Review Workflow using LangGraph

This module implements a generic two-stage review workflow where:
1. A ReAct agent produces initial output
2. A review agent evaluates the output quality
3. If approved, the workflow finishes; if rejected, it retries up to max_retries
"""

from typing import List, Optional, Literal, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import operator


class AgentState(TypedDict):
    """State schema for the two-stage review workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    current_output: str
    review_result: str
    retry_count: int
    max_retries: int


def create_default_react_agent() -> AgentExecutor:
    """Create a default ReAct agent with basic tools"""
    
    # Simple search tool (mock implementation for demo)
    def search_tool(query: str) -> str:
        """A mock search tool that returns a simple response"""
        return f"Search results for '{query}': This is a mock search result providing relevant information about the query."
    
    # Calculator tool
    def calculator_tool(expression: str) -> str:
        """A simple calculator tool"""
        try:
            # Basic safety check - only allow simple mathematical expressions
            allowed_chars = set('0123456789+-*/().')
            if all(c in allowed_chars or c.isspace() for c in expression):
                result = eval(expression)
                return f"The result of {expression} is {result}"
            else:
                return "Invalid expression. Only basic mathematical operations are allowed."
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    tools = [
        Tool(
            name="search",
            description="Search for information on the internet",
            func=search_tool
        ),
        Tool(
            name="calculator",
            description="Perform mathematical calculations",
            func=calculator_tool
        )
    ]
    
    # Create ReAct prompt template
    react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
    
    # Create the ReAct agent
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    agent = create_react_agent(llm, tools, react_prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)


def react_agent_node(state: AgentState, react_agent: Optional[AgentExecutor] = None) -> AgentState:
    """
    Execute the ReAct agent and update state with the output
    
    Args:
        state: Current workflow state
        react_agent: Optional ReAct agent instance (uses default if None)
    
    Returns:
        Updated state with agent output
    """
    if react_agent is None:
        react_agent = create_default_react_agent()
    
    # Get the latest human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        raise ValueError("No human message found in state")
    
    latest_question = human_messages[-1].content
    
    # Execute the ReAct agent
    try:
        result = react_agent.invoke({"input": latest_question})
        output = result.get("output", "No output generated")
    except Exception as e:
        output = f"Error executing ReAct agent: {str(e)}"
    
    # Update state
    return {
        **state,
        "current_output": output,
        "retry_count": state.get("retry_count", 0) + 1,
        "messages": state["messages"] + [AIMessage(content=f"ReAct Agent Output: {output}")]
    }


def review_agent_node(state: AgentState) -> AgentState:
    """
    Review the ReAct agent's output for quality and completeness
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with review result
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Get the original question
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    original_question = human_messages[-1].content if human_messages else "Unknown question"
    
    current_output = state.get("current_output", "")
    
    # Create review prompt
    review_prompt = f"""
You are a quality reviewer for AI agent outputs. Your job is to evaluate whether the agent's response adequately addresses the user's question.

Original Question: {original_question}

Agent's Output: {current_output}

Please evaluate the output based on these criteria:
1. Does it directly address the original question?
2. Is the information factually sound and reasonable?
3. Is the response complete and helpful?
4. Are there any obvious errors or omissions?

Respond with either "APPROVED" or "REJECTED" followed by a brief explanation.

If the output is generally good and addresses the question reasonably well, approve it.
If the output is clearly inadequate, contains errors, or doesn't address the question, reject it.

Your response:
"""
    
    try:
        review_response = llm.invoke([HumanMessage(content=review_prompt)])
        review_content = review_response.content.strip()
        
        # Determine if approved or rejected
        if review_content.upper().startswith("APPROVED"):
            review_result = "approved"
        elif review_content.upper().startswith("REJECTED"):
            review_result = "rejected"
        else:
            # Default to rejected if unclear
            review_result = "rejected"
            review_content = f"REJECTED - Unclear review response: {review_content}"
        
    except Exception as e:
        review_result = "rejected"
        review_content = f"REJECTED - Error during review: {str(e)}"
    
    return {
        **state,
        "review_result": review_result,
        "messages": state["messages"] + [AIMessage(content=f"Review Result: {review_content}")]
    }


def should_continue(state: AgentState) -> Literal["react_agent", "END"]:
    """
    Determine the next step based on review result and retry count
    
    Args:
        state: Current workflow state
    
    Returns:
        Next node to execute or END
    """
    review_result = state.get("review_result", "")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    # If approved, end the workflow
    if review_result == "approved":
        return "END"
    
    # If max retries reached, end the workflow
    if retry_count >= max_retries:
        return "END"
    
    # Otherwise, retry with the ReAct agent
    return "react_agent"


# Create the workflow graph
def create_workflow() -> StateGraph:
    """Create and configure the two-stage review workflow"""
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    
    # Set entry point
    workflow.set_entry_point("react_agent")
    
    # Add edges
    workflow.add_edge("react_agent", "review_agent")
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "react_agent": "react_agent",
            "END": END
        }
    )
    
    return workflow


# Create and compile the graph
workflow = create_workflow()
app = workflow.compile()


# Helper function to create initial state
def create_initial_state(messages: List[BaseMessage], max_retries: int = 3) -> AgentState:
    """
    Create initial state for the workflow
    
    Args:
        messages: List of messages (should contain at least one HumanMessage)
        max_retries: Maximum number of retry attempts
    
    Returns:
        Initial state dictionary
    """
    return {
        "messages": messages,
        "current_output": "",
        "review_result": "",
        "retry_count": 0,
        "max_retries": max_retries
    }


# Example usage function
def run_workflow_example():
    """Example of how to use the workflow"""
    initial_state = create_initial_state([
        HumanMessage("What is the capital of France and what is 15 + 27?")
    ])
    
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Run example if script is executed directly
    result = run_workflow_example()
    print("Final result:", result)

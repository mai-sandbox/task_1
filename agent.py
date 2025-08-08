"""
Generic Review Agent System with ReAct Agent

This system orchestrates a ReAct agent with a review mechanism that evaluates
outputs and triggers re-execution if needed.
"""

from typing import Dict, Any, List, Optional, Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.graph.message import add_messages
import operator
import os

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class MockLLM:
    """Mock LLM for testing when OpenAI API key is not available"""
    
    def invoke(self, messages, **kwargs):
        """Mock invoke method that returns predictable responses"""
        if isinstance(messages, list) and len(messages) > 0:
            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            if "Review" in content and "APPROVED" in content:
                return AIMessage(content="APPROVED")
            elif "Review" in content:
                return AIMessage(content="APPROVED")
            else:
                return AIMessage(content="Mock response: Task completed successfully.")
        
        return AIMessage(content="Mock response")


class AgentState(TypedDict):
    """State for the orchestrator agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: str
    react_output: Optional[str]
    review_feedback: Optional[str]
    iteration_count: int
    max_iterations: int
    is_approved: bool


class ReviewState(TypedDict):
    """State for the review agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    output_to_review: str
    feedback: Optional[str]
    is_approved: bool


@tool
def example_calculator(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def example_text_analyzer(text: str) -> str:
    """Analyze text and return basic statistics"""
    words = text.split()
    return f"Text has {len(words)} words and {len(text)} characters"


def create_react_agent_node(tools: List = None, llm: Optional[Any] = None):
    """Create a configurable ReAct agent"""
    if tools is None:
        tools = [example_calculator, example_text_analyzer]
    
    if llm is None:
        if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
            return create_mock_react_agent_node(tools)
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    react_graph = create_react_agent(llm, tools)
    
    def react_agent_node(state: AgentState) -> Dict[str, Any]:
        """Execute the ReAct agent"""
        task_message = HumanMessage(content=state["task"])
        
        if state.get("review_feedback"):
            feedback_message = SystemMessage(
                content=f"Previous attempt feedback: {state['review_feedback']}. Please address this feedback."
            )
            messages = [feedback_message, task_message]
        else:
            messages = [task_message]
        
        result = react_graph.invoke({"messages": messages})
        
        last_message = result["messages"][-1]
        output = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        return {
            "react_output": output,
            "messages": [AIMessage(content=f"ReAct output: {output}")],
            "iteration_count": state["iteration_count"] + 1
        }
    
    return react_agent_node


def create_mock_react_agent_node(tools: List):
    """Create a mock ReAct agent for testing"""
    def mock_react_agent_node(state: AgentState) -> Dict[str, Any]:
        """Mock ReAct agent implementation"""
        task = state["task"]
        
        if "calculate" in task.lower() or "math" in task.lower():
            output = "I calculated the result using the calculator tool. The answer is 42."
        elif "analyze" in task.lower():
            output = "I analyzed the text using the text analyzer tool. The text contains important information."
        else:
            output = f"I completed the task: {task}"
        
        if state.get("review_feedback"):
            output = f"Revised: {output} (addressed feedback: {state['review_feedback']})"
        
        return {
            "react_output": output,
            "messages": [AIMessage(content=f"ReAct output: {output}")],
            "iteration_count": state["iteration_count"] + 1
        }
    
    return mock_react_agent_node


def create_review_agent_node(llm: Optional[Any] = None):
    """Create a review agent that evaluates ReAct outputs"""
    if llm is None:
        if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
            llm = MockLLM()
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def review_agent_node(state: AgentState) -> Dict[str, Any]:
        """Review the ReAct agent's output"""
        review_prompt = f"""
        Review the following output from a ReAct agent for the task: "{state['task']}"
        
        Output to review:
        {state['react_output']}
        
        Evaluate if this output:
        1. Correctly addresses the task
        2. Is complete and accurate
        3. Is well-formatted and clear
        
        Respond with:
        - "APPROVED" if the output is satisfactory
        - "NEEDS_REVISION: [specific feedback]" if improvements are needed
        
        Be constructive and specific in your feedback.
        """
        
        review_message = HumanMessage(content=review_prompt)
        response = llm.invoke([SystemMessage(content="You are a helpful review agent."), review_message])
        
        review_content = response.content
        is_approved = "APPROVED" in review_content
        
        feedback = None if is_approved else review_content.replace("NEEDS_REVISION:", "").strip()
        
        return {
            "review_feedback": feedback,
            "is_approved": is_approved,
            "messages": [AIMessage(content=f"Review: {review_content}")]
        }
    
    return review_agent_node


def should_continue(state: AgentState) -> str:
    """Determine if we should continue iterating or finish"""
    if state["is_approved"]:
        return "finish"
    
    if state["iteration_count"] >= state["max_iterations"]:
        return "finish"
    
    return "iterate"


def finish_node(state: AgentState) -> Dict[str, Any]:
    """Finalize the process"""
    if state["is_approved"]:
        final_message = f"Task completed successfully!\n\nFinal output:\n{state['react_output']}"
    else:
        final_message = f"Maximum iterations reached. Last output:\n{state['react_output']}\n\nLast feedback:\n{state.get('review_feedback', 'None')}"
    
    return {
        "messages": [AIMessage(content=final_message)]
    }


def build_review_agent_system(
    tools: Optional[List] = None,
    react_llm: Optional[Any] = None,
    review_llm: Optional[Any] = None,
    max_iterations: int = 3
) -> StateGraph:
    """
    Build the complete review agent system
    
    Args:
        tools: List of tools for the ReAct agent
        react_llm: LLM for the ReAct agent
        review_llm: LLM for the review agent
        max_iterations: Maximum number of review iterations
    
    Returns:
        Compiled LangGraph application
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("react_agent", create_react_agent_node(tools, react_llm))
    workflow.add_node("review_agent", create_review_agent_node(review_llm))
    workflow.add_node("finish", finish_node)
    
    workflow.set_entry_point("react_agent")
    
    workflow.add_edge("react_agent", "review_agent")
    
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "iterate": "react_agent",
            "finish": "finish"
        }
    )
    
    workflow.add_edge("finish", END)
    
    return workflow.compile()


app = build_review_agent_system(max_iterations=3)


if __name__ == "__main__":
    test_task = "Calculate the sum of 125 + 375 and then analyze the result text"
    
    initial_state = {
        "messages": [],
        "task": test_task,
        "react_output": None,
        "review_feedback": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "is_approved": False
    }
    
    print(f"Running task: {test_task}\n")
    print("=" * 50)
    
    for event in app.stream(initial_state):
        for node, data in event.items():
            print(f"\nNode: {node}")
            if "messages" in data and data["messages"]:
                for msg in data["messages"]:
                    print(f"  {msg.content}")
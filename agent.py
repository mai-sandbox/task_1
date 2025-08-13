from typing import TypedDict, List, Any, Callable
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class AgentState(TypedDict):
    messages: List[Any]
    current_output: str
    review_result: str
    iterations: int
    max_iterations: int
    react_agent: Callable

def run_react_agent(state: AgentState) -> AgentState:
    react_agent = state.get("react_agent")
    if not react_agent:
        raise ValueError("No react_agent provided in state")
    
    messages = state["messages"]
    result = react_agent.invoke({"messages": messages})
    
    if hasattr(result, 'content'):
        output = result.content
    elif isinstance(result, dict) and 'messages' in result:
        last_message = result['messages'][-1]
        output = last_message.content if hasattr(last_message, 'content') else str(last_message)
    else:
        output = str(result)
    
    return {
        **state,
        "current_output": output,
        "iterations": state.get("iterations", 0) + 1
    }

def review_output(state: AgentState) -> AgentState:
    current_output = state["current_output"]
    original_message = state["messages"][0].content if state["messages"] else "No original message"
    
    review_prompt = f"""
    Review the following output for quality and correctness:
    
    Original Request: {original_message}
    
    Agent Output: {current_output}
    
    Is this output satisfactory? Respond with either:
    - "APPROVED: [brief reason]" if the output is good
    - "NEEDS_IMPROVEMENT: [specific feedback]" if it needs work
    """
    
    review_result = f"APPROVED: Output appears complete and relevant to the request"
    
    if len(current_output.strip()) < 10:
        review_result = "NEEDS_IMPROVEMENT: Output is too brief and lacks detail"
    elif "error" in current_output.lower() or "sorry" in current_output.lower():
        review_result = "NEEDS_IMPROVEMENT: Output indicates an error or inability to complete the task"
    
    return {
        **state,
        "review_result": review_result
    }

def should_continue(state: AgentState) -> str:
    review_result = state.get("review_result", "")
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if iterations >= max_iterations:
        return "finish"
    
    if review_result.startswith("APPROVED"):
        return "finish"
    else:
        return "retry"

def prepare_retry(state: AgentState) -> AgentState:
    review_feedback = state.get("review_result", "")
    original_messages = state["messages"]
    
    retry_message = HumanMessage(
        content=f"{original_messages[0].content}\n\nPrevious attempt feedback: {review_feedback}\nPlease improve your response."
    )
    
    return {
        **state,
        "messages": [retry_message]
    }

def finalize_output(state: AgentState) -> AgentState:
    final_message = AIMessage(content=state["current_output"])
    
    return {
        **state,
        "messages": state["messages"] + [final_message]
    }

workflow = StateGraph(AgentState)

workflow.add_node("react_agent", run_react_agent)
workflow.add_node("reviewer", review_output)
workflow.add_node("retry_prep", prepare_retry)
workflow.add_node("finalize", finalize_output)

workflow.set_entry_point("react_agent")

workflow.add_edge("react_agent", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "finish": "finalize",
        "retry": "retry_prep"
    }
)

workflow.add_edge("retry_prep", "react_agent")
workflow.add_edge("finalize", END)

app = workflow.compile()
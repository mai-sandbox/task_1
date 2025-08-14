"""
Dual Agent System with React Agent and Review Agent
"""
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]
    current_output: Optional[str]
    review_feedback: Optional[str]
    is_approved: bool
    iteration_count: int
    max_iterations: int
    react_agent: Optional[callable]


def create_dual_agent_system(react_agent_func: callable = None, max_iterations: int = 3):
    """
    Creates a dual agent system with a React agent and Review agent.
    
    Args:
        react_agent_func: Function that takes messages and returns a response
        max_iterations: Maximum number of iterations before forcing completion
    """
    
    def default_react_agent(messages: List) -> str:
        """Default React agent implementation"""
        if not messages:
            return "No input provided."
        
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            # Simple React-style response
            return f"I'll help you with: {last_message.content}. Let me think about this step by step:\n\n1. Understanding the request\n2. Processing the information\n3. Providing a helpful response\n\nBased on your request, here's my response: This is a sample React agent response that processes your input and provides a structured answer."
        return "I need a human message to process."
    
    def react_agent_node(state: AgentState) -> AgentState:
        """React agent node that generates initial output or processes feedback"""
        agent_func = state.get('react_agent') or default_react_agent
        
        # If there's review feedback, incorporate it into the messages
        messages = state['messages'].copy()
        if state.get('review_feedback') and state['iteration_count'] > 0:
            feedback_message = SystemMessage(
                content=f"Previous attempt was not approved. Reviewer feedback: {state['review_feedback']}. Please improve your response."
            )
            messages.append(feedback_message)
        
        # Generate response using the React agent
        output = agent_func(messages)
        
        return {
            **state,
            'current_output': output,
            'iteration_count': state['iteration_count'] + 1
        }
    
    def review_agent_node(state: AgentState) -> AgentState:
        """Review agent node that evaluates the React agent's output"""
        current_output = state['current_output']
        
        if not current_output:
            return {
                **state,
                'is_approved': False,
                'review_feedback': "No output to review."
            }
        
        # Simple review logic - can be made more sophisticated
        quality_indicators = [
            len(current_output) > 50,  # Sufficient length
            '?' not in current_output or current_output.count('?') <= 2,  # Not too many questions
            any(word in current_output.lower() for word in ['help', 'answer', 'response', 'solution']),  # Contains helpful content
            not any(word in current_output.lower() for word in ['error', 'failed', 'cannot', "don't know"])  # No obvious failures
        ]
        
        score = sum(quality_indicators)
        is_approved = score >= 3  # Require at least 3 out of 4 quality indicators
        
        if is_approved:
            feedback = "Output approved. Good quality response with sufficient detail and helpful content."
        else:
            feedback_issues = []
            if len(current_output) <= 50:
                feedback_issues.append("Response is too short")
            if current_output.count('?') > 2:
                feedback_issues.append("Too many questions, provide more definitive answers")
            if not any(word in current_output.lower() for word in ['help', 'answer', 'response', 'solution']):
                feedback_issues.append("Response lacks helpful content")
            if any(word in current_output.lower() for word in ['error', 'failed', 'cannot', "don't know"]):
                feedback_issues.append("Response contains failure indicators")
            
            feedback = "Output needs improvement: " + "; ".join(feedback_issues)
        
        return {
            **state,
            'is_approved': is_approved,
            'review_feedback': feedback
        }
    
    def should_continue(state: AgentState) -> str:
        """Decide whether to continue the loop or end"""
        if state['is_approved']:
            return "end"
        elif state['iteration_count'] >= state['max_iterations']:
            return "end"
        else:
            return "continue"
    
    def finalize_output(state: AgentState) -> AgentState:
        """Finalize the output with the final response"""
        if state['is_approved']:
            final_message = AIMessage(content=state['current_output'])
        else:
            final_message = AIMessage(
                content=f"Maximum iterations ({state['max_iterations']}) reached. Best attempt:\n\n{state['current_output']}\n\nFinal feedback: {state['review_feedback']}"
            )
        
        return {
            **state,
            'messages': state['messages'] + [final_message]
        }
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("react_agent", react_agent_node)
    workflow.add_node("review_agent", review_agent_node)
    workflow.add_node("finalize", finalize_output)
    
    # Set entry point
    workflow.set_entry_point("react_agent")
    
    # Add edges
    workflow.add_edge("react_agent", "review_agent")
    workflow.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "continue": "react_agent",
            "end": "finalize"
        }
    )
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# Create the default app instance
app = create_dual_agent_system()


def create_custom_dual_agent(react_agent_func: callable, max_iterations: int = 3):
    """
    Helper function to create a custom dual agent system with a specific React agent.
    
    Args:
        react_agent_func: Custom React agent function
        max_iterations: Maximum number of iterations
        
    Returns:
        Compiled LangGraph application
    """
    return create_dual_agent_system(react_agent_func, max_iterations)
"""
Generic two-agent system with React agent and review agent.
The React agent processes the initial request, and the review agent
evaluates the output. If the review is negative, it sends back to React agent.
"""

from typing import TypedDict, Literal, Annotated, Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import operator

# State definition
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    react_output: str
    review_result: str
    review_approved: bool
    iteration_count: int
    max_iterations: int

# Default tools for React agent (can be customized)
@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 75°F"

@tool
def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression."""
    try:
        # Simple calculation - in real use, would use safer eval
        result = eval(expression.replace("^", "**"))
        return str(result)
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

# Default tools list (can be overridden)
DEFAULT_TOOLS = [get_current_weather, calculate]

def create_react_node(tools=None, model_name="gpt-4"):
    """Create a React agent node with configurable tools and model."""
    if tools is None:
        tools = DEFAULT_TOOLS
    
    llm = ChatOpenAI(model=model_name, temperature=0)
    react_agent = create_react_agent(llm, tools)
    
    def react_node(state: AgentState):
        """Process the user request using React agent."""
        # Get the original human message
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not human_messages:
            return {"react_output": "No user message found"}
        
        # Build messages for react agent
        messages = [human_messages[-1]]
        
        # If there's previous review feedback, include it
        if state.get("review_result") and not state.get("review_approved", False):
            feedback_msg = HumanMessage(
                content=f"Previous attempt was rejected. Review feedback: {state['review_result']}. "
                       f"Please improve your response based on this feedback."
            )
            messages.append(feedback_msg)
        
        # Create input for react agent
        react_input = {"messages": messages}
        
        # Run the react agent
        result = react_agent.invoke(react_input)
        
        # Extract the final AI response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        react_output = ai_messages[-1].content if ai_messages else "No response from React agent"
        
        return {
            "react_output": react_output,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    
    return react_node

def create_review_node(model_name="gpt-4", review_criteria=None):
    """Create a review agent node with configurable model and criteria."""
    if review_criteria is None:
        review_criteria = """
        Evaluate the response based on:
        1. Accuracy and correctness
        2. Completeness of the answer
        3. Clarity and helpfulness
        4. Appropriate use of available tools
        
        Respond with 'APPROVED' if the response meets these criteria, 
        or 'REJECTED: [reason]' if it needs improvement.
        """
    
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    def review_node(state: AgentState):
        """Review the React agent's output."""
        react_output = state.get("react_output", "")
        original_question = ""
        
        # Get original human message for context
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if human_messages:
            original_question = human_messages[-1].content
        
        review_prompt = f"""
        Original question: {original_question}
        
        React agent response: {react_output}
        
        {review_criteria}
        """
        
        messages = [SystemMessage(content="You are a quality reviewer for AI responses."),
                   HumanMessage(content=review_prompt)]
        
        result = llm.invoke(messages)
        review_result = result.content
        
        # Determine if approved
        approved = review_result.upper().startswith("APPROVED")
        
        return {
            "review_result": review_result,
            "review_approved": approved
        }
    
    return review_node

def should_continue_from_react(state: AgentState) -> Literal["review", "end"]:
    """Determine next step from React agent."""
    # If max iterations reached, end
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return "end"
    
    # Otherwise go to review
    return "review"

def should_continue_from_review(state: AgentState) -> Literal["react", "end"]:
    """Determine next step from Review agent."""
    # If review approved, end
    if state.get("review_approved", False):
        return "end"
    
    # If max iterations reached, end
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return "end"
    
    # Otherwise, send back to React agent for improvement
    return "react"

def create_final_response_node():
    """Create final response based on the process results."""
    def final_response_node(state: AgentState):
        """Generate final response message."""
        if state.get("review_approved", False):
            final_message = AIMessage(content=state["react_output"])
        else:
            final_message = AIMessage(
                content=f"After {state.get('max_iterations', 3)} iterations, "
                       f"the best response is: {state['react_output']}\n\n"
                       f"Final review: {state.get('review_result', 'No review available')}"
            )
        
        return {"messages": [final_message]}
    
    return final_response_node

def create_dual_agent_graph(
    react_tools=None, 
    react_model="gpt-4",
    review_model="gpt-4", 
    review_criteria=None,
    max_iterations=3
):
    """
    Create a dual-agent system with React and Review agents.
    
    Args:
        react_tools: List of tools for React agent (defaults to built-in tools)
        react_model: Model name for React agent
        review_model: Model name for Review agent  
        review_criteria: Custom review criteria string
        max_iterations: Maximum number of React->Review cycles
    """
    
    # Create nodes
    react_node = create_react_node(react_tools, react_model)
    review_node = create_review_node(review_model, review_criteria)
    final_response_node = create_final_response_node()
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("react", react_node)
    workflow.add_node("review", review_node)
    workflow.add_node("final", final_response_node)
    
    # Set entry point
    workflow.add_edge(START, "react")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "react",
        should_continue_from_react,
        {
            "review": "review", 
            "end": "final"
        }
    )
    
    workflow.add_conditional_edges(
        "review", 
        should_continue_from_review,
        {
            "react": "react",
            "end": "final"
        }
    )
    
    # Final node goes to END
    workflow.add_edge("final", END)
    
    # Compile and return
    graph = workflow.compile()
    
    # Save original invoke method
    original_invoke = graph.invoke
    
    def invoke_with_defaults(input_dict):
        """Invoke the graph with default max_iterations."""
        if isinstance(input_dict, dict) and "messages" in input_dict:
            # Add default max_iterations if not specified
            if "max_iterations" not in input_dict:
                input_dict["max_iterations"] = max_iterations
            if "iteration_count" not in input_dict:
                input_dict["iteration_count"] = 0
        # Call the original invoke method
        return original_invoke(input_dict)
    
    # Replace the graph's invoke method
    graph.invoke = invoke_with_defaults
    
    return graph

# Create default app export
app = create_dual_agent_graph()
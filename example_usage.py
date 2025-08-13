"""
Example usage of the dual-agent system with custom React agents.
This demonstrates how to integrate any React agent into the review workflow.
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agent import create_app


# Example 1: Research-focused React agent
@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    # This would normally call a real search API
    return f"Search results for '{query}': Found relevant information about {query}."

@tool
def summarize_tool(text: str) -> str:
    """Summarize a given text."""
    return f"Summary: {text[:100]}..."

def create_research_agent():
    """Create a React agent focused on research tasks."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [web_search_tool, summarize_tool]
    return create_react_agent(llm, tools)


# Example 2: Math-focused React agent
@tool
def advanced_calculator(expression: str) -> str:
    """Calculate complex mathematical expressions."""
    try:
        # Simple eval for demo - in production, use a proper math parser
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between different units."""
    # Simple demo conversions
    conversions = {
        ("meters", "feet"): 3.28084,
        ("celsius", "fahrenheit"): lambda c: c * 9/5 + 32,
        ("kg", "lbs"): 2.20462
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        factor = conversions[key]
        if callable(factor):
            result = factor(value)
        else:
            result = value * factor
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    else:
        return f"Conversion from {from_unit} to {to_unit} not supported"

def create_math_agent():
    """Create a React agent focused on mathematical tasks."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [advanced_calculator, unit_converter]
    return create_react_agent(llm, tools)


def demo_research_agent():
    """Demonstrate the dual-agent system with a research-focused React agent."""
    print("=== Research Agent Demo ===")
    
    research_agent = create_research_agent()
    research_app = create_app(research_agent, max_iterations=3)
    
    query = "What are the latest developments in quantum computing?"
    print(f"Query: {query}")
    print("-" * 50)
    
    result = research_app.invoke({
        "messages": [HumanMessage(query)],
        "max_iterations": 3,
        "iteration_count": 0,
        "is_approved": False,
        "react_output": None,
        "review_feedback": None,
        "react_agent_graph": None
    })
    
    if result and "messages" in result:
        final_message = result["messages"][-1]
        print(f"Final Answer: {final_message.content}")
    
    print(f"Iterations used: {result.get('iteration_count')}")
    print(f"Approved: {result.get('is_approved')}")
    print()


def demo_math_agent():
    """Demonstrate the dual-agent system with a math-focused React agent."""
    print("=== Math Agent Demo ===")
    
    math_agent = create_math_agent()
    math_app = create_app(math_agent, max_iterations=3)
    
    query = "Convert 100 meters to feet, then calculate what 25% of that value would be"
    print(f"Query: {query}")
    print("-" * 50)
    
    result = math_app.invoke({
        "messages": [HumanMessage(query)],
        "max_iterations": 3,
        "iteration_count": 0,
        "is_approved": False,
        "react_output": None,
        "review_feedback": None,
        "react_agent_graph": None
    })
    
    if result and "messages" in result:
        final_message = result["messages"][-1]
        print(f"Final Answer: {final_message.content}")
    
    print(f"Iterations used: {result.get('iteration_count')}")
    print(f"Approved: {result.get('is_approved')}")
    print()


def demo_custom_integration():
    """Show how to integrate any existing React agent."""
    print("=== Custom Integration Demo ===")
    print("This shows how you can take ANY React agent and wrap it with the reviewer:")
    print()
    
    # Imagine you have an existing React agent from somewhere else
    existing_react_agent = create_research_agent()  # This could be any React agent
    
    # Simply wrap it with the dual-agent system
    enhanced_agent = create_app(
        react_agent=existing_react_agent,
        max_iterations=2  # Customize retry behavior
    )
    
    # Use it exactly like any other LangGraph app
    result = enhanced_agent.invoke({
        "messages": [HumanMessage("Explain machine learning in simple terms")]
    })
    
    print("Integration complete! The agent now includes:")
    print("✓ Original React agent functionality")
    print("✓ Automatic output review")
    print("✓ Iterative improvement")
    print("✓ Quality assurance")
    print()


if __name__ == "__main__":
    print("Dual-Agent System Usage Examples")
    print("="*50)
    print()
    
    # Note: These demos would normally require API keys
    print("Note: These examples require OpenAI API key to be set in environment")
    print("For demonstration purposes, showing the structure...")
    print()
    
    demo_custom_integration()
    
    # Uncomment these if you have API keys configured:
    # demo_research_agent()
    # demo_math_agent()
    
    print("Key Benefits of the Dual-Agent System:")
    print("• Generic: Works with any React agent")
    print("• Quality-focused: Automatic review and improvement")
    print("• Configurable: Adjustable retry limits and criteria")
    print("• LangGraph native: Full integration with LangGraph ecosystem")
    print("• Deployment ready: Includes langgraph.json configuration")
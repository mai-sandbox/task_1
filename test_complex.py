"""
Test the dual-agent system with a more complex query that might require iterations.
"""

from langchain_core.messages import HumanMessage
from agent import app, create_app, create_default_react_agent


def test_complex_query():
    """Test with a query that might need multiple iterations."""
    
    initial_state = {
        "messages": [HumanMessage("Explain the theory of relativity and provide a practical example of its effects.")],
        "max_iterations": 3,
        "iteration_count": 0,
        "is_approved": False,
        "react_output": None,
        "review_feedback": None,
        "react_agent_graph": None
    }
    
    try:
        print("Testing dual-agent system with complex query...")
        print("Input query: Explain the theory of relativity and provide a practical example of its effects.")
        print("-" * 80)
        
        result = app.invoke(initial_state)
        
        print("Final result:")
        if result and "messages" in result:
            for i, message in enumerate(result["messages"]):
                print(f"Message {i+1} ({type(message).__name__}):")
                print(f"  {message.content[:200]}{'...' if len(message.content) > 200 else ''}")
                print()
        
        print(f"Total iterations: {result.get('iteration_count', 'Unknown')}")
        print(f"Final approval status: {result.get('is_approved', 'Unknown')}")
        
        if result.get("review_feedback"):
            print(f"\nFinal review feedback:")
            print(f"  {result['review_feedback'][:300]}{'...' if len(result['review_feedback']) > 300 else ''}")
            
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_custom_react_agent():
    """Test with a custom React agent to show flexibility."""
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    
    @tool
    def weather_tool(location: str) -> str:
        """Get weather information for a location."""
        return f"The weather in {location} is sunny with a temperature of 72°F."
    
    @tool
    def time_tool(timezone: str = "UTC") -> str:
        """Get current time for a timezone."""
        return f"The current time in {timezone} is 2:30 PM."
    
    # Create custom React agent with different tools
    custom_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    custom_tools = [weather_tool, time_tool]
    custom_react_agent = create_react_agent(custom_llm, custom_tools)
    
    # Create dual-agent system with custom React agent
    custom_app = create_app(custom_react_agent, max_iterations=2)
    
    initial_state = {
        "messages": [HumanMessage("What's the weather like in New York and what time is it there?")],
        "max_iterations": 2,
        "iteration_count": 0,
        "is_approved": False,
        "react_output": None,
        "review_feedback": None,
        "react_agent_graph": None
    }
    
    try:
        print("\n" + "="*80)
        print("Testing with custom React agent...")
        print("Input query: What's the weather like in New York and what time is it there?")
        print("-" * 80)
        
        result = custom_app.invoke(initial_state)
        
        print("Final result:")
        if result and "messages" in result:
            for i, message in enumerate(result["messages"]):
                print(f"Message {i+1} ({type(message).__name__}):")
                print(f"  {message.content}")
                print()
        
        print(f"Total iterations: {result.get('iteration_count', 'Unknown')}")
        print(f"Final approval status: {result.get('is_approved', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"Error during custom agent test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running complex query test...")
    success1 = test_complex_query()
    
    print("\nRunning custom agent test...")
    success2 = test_with_custom_react_agent()
    
    if success1 and success2:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
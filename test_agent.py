"""
Simple test script for the dual-agent system.
"""

from langchain_core.messages import HumanMessage
from agent import app


def test_basic_functionality():
    """Test basic functionality of the dual-agent system."""
    
    # Test input
    initial_state = {
        "messages": [HumanMessage("What is 2 + 2?")],
        "max_iterations": 3,
        "iteration_count": 0,
        "is_approved": False,
        "react_output": None,
        "review_feedback": None,
        "react_agent_graph": None
    }
    
    try:
        print("Testing dual-agent system...")
        print("Input query: What is 2 + 2?")
        print("-" * 50)
        
        # Run the agent
        result = app.invoke(initial_state)
        
        print("Final result:")
        if result and "messages" in result:
            for message in result["messages"]:
                print(f"- {type(message).__name__}: {message.content}")
        
        print(f"\nIteration count: {result.get('iteration_count', 'Unknown')}")
        print(f"Final approval status: {result.get('is_approved', 'Unknown')}")
        
        if result.get("review_feedback"):
            print(f"Final review feedback: {result['review_feedback']}")
            
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
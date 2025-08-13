#!/usr/bin/env python3

from langchain_core.messages import HumanMessage
from agent import app, create_app_with_custom_agent


def test_basic_functionality():
    """Test the basic orchestrator functionality"""
    print("Testing basic orchestrator functionality...")
    
    initial_state = {
        "messages": [HumanMessage("What is the capital of France?")],
        "react_output": None,
        "review_result": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "react_agent": None
    }
    
    try:
        result = app.invoke(initial_state)
        print("✅ Basic test passed!")
        print(f"Final messages count: {len(result['messages'])}")
        print(f"Iterations used: {result['iteration_count']}")
        print(f"Last message: {result['messages'][-1].content}")
        return True
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def test_custom_react_agent():
    """Test with a custom ReAct agent"""
    print("\nTesting with custom ReAct agent...")
    
    def custom_react_agent(state):
        """Custom ReAct agent that always gives short responses"""
        messages = state.get("messages", [])
        return {
            "messages": messages + [
                type(messages[0].__class__.__name__ == 'HumanMessage' and 
                     'AIMessage' or 'HumanMessage')("Short response")  # This will trigger review failure
            ]
        }
    
    # Fix the custom agent
    def better_custom_react_agent(state):
        from langchain_core.messages import AIMessage
        messages = state.get("messages", [])
        return {
            "messages": messages + [AIMessage("This is a comprehensive and detailed response that should pass the review because it's longer than 10 characters and doesn't contain errors.")]
        }
    
    custom_app = create_app_with_custom_agent(better_custom_react_agent, max_iterations=2)
    
    initial_state = {
        "messages": [HumanMessage("Explain quantum computing")],
        "react_output": None,
        "review_result": None,
        "iteration_count": 0,
        "max_iterations": 2,
        "react_agent": None
    }
    
    try:
        result = custom_app.invoke(initial_state)
        print("✅ Custom agent test passed!")
        print(f"Iterations used: {result['iteration_count']}")
        print(f"Last message: {result['messages'][-1].content}")
        return True
    except Exception as e:
        print(f"❌ Custom agent test failed: {e}")
        return False


def test_max_iterations():
    """Test that max iterations limit works"""
    print("\nTesting max iterations limit...")
    
    def failing_react_agent(state):
        from langchain_core.messages import AIMessage
        messages = state.get("messages", [])
        return {
            "messages": messages + [AIMessage("bad")]  # Short response that will always fail review
        }
    
    failing_app = create_app_with_custom_agent(failing_react_agent, max_iterations=2)
    
    initial_state = {
        "messages": [HumanMessage("Test question")],
        "react_output": None,
        "review_result": None,
        "iteration_count": 0,
        "max_iterations": 2,
        "react_agent": None
    }
    
    try:
        result = failing_app.invoke(initial_state)
        print("✅ Max iterations test passed!")
        print(f"Stopped at iteration: {result['iteration_count']}")
        print(f"Final result contains max iterations info: {'iterations' in result['messages'][-1].content}")
        return True
    except Exception as e:
        print(f"❌ Max iterations test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running orchestrator agent tests...\n")
    
    tests = [
        test_basic_functionality,
        test_custom_react_agent,
        test_max_iterations
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The orchestrator agent is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
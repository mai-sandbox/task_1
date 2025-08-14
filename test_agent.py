"""
Test script for the dual agent system
"""
from agent import app, create_custom_dual_agent
from langchain_core.messages import HumanMessage


def test_basic_functionality():
    """Test the basic functionality of the dual agent system"""
    print("=== Testing Basic Functionality ===")
    
    # Test with a simple message
    initial_state = {
        "messages": [HumanMessage("Hello, can you help me understand Python functions?")],
        "current_output": None,
        "review_feedback": None,
        "is_approved": False,
        "iteration_count": 0,
        "max_iterations": 3,
        "react_agent": None
    }
    
    result = app.invoke(initial_state)
    
    print(f"Final messages count: {len(result['messages'])}")
    print(f"Iteration count: {result['iteration_count']}")
    print(f"Was approved: {result['is_approved']}")
    print(f"Final output: {result['current_output'][:100]}...")
    print(f"Review feedback: {result['review_feedback']}")
    print()


def test_custom_react_agent():
    """Test with a custom React agent"""
    print("=== Testing Custom React Agent ===")
    
    def custom_react_agent(messages):
        """Custom React agent that gives short responses (should trigger review failure)"""
        return "Yes."
    
    custom_app = create_custom_dual_agent(custom_react_agent, max_iterations=2)
    
    initial_state = {
        "messages": [HumanMessage("Explain machine learning to me.")],
        "current_output": None,
        "review_feedback": None,
        "is_approved": False,
        "iteration_count": 0,
        "max_iterations": 2,
        "react_agent": custom_react_agent
    }
    
    result = custom_app.invoke(initial_state)
    
    print(f"Final messages count: {len(result['messages'])}")
    print(f"Iteration count: {result['iteration_count']}")
    print(f"Was approved: {result['is_approved']}")
    print(f"Final output: {result['current_output']}")
    print(f"Review feedback: {result['review_feedback']}")
    print()


def test_good_custom_agent():
    """Test with a custom React agent that should pass review"""
    print("=== Testing Good Custom React Agent ===")
    
    def good_react_agent(messages):
        """Custom React agent that gives comprehensive responses"""
        if messages and len(messages) > 0:
            last_msg = messages[-1]
            return f"I'll help you with {last_msg.content}. Here's a comprehensive answer with detailed explanation and helpful information to address your question thoroughly."
        return "I'm ready to help you with any questions you have."
    
    good_app = create_custom_dual_agent(good_react_agent, max_iterations=3)
    
    initial_state = {
        "messages": [HumanMessage("What is artificial intelligence?")],
        "current_output": None,
        "review_feedback": None,
        "is_approved": False,
        "iteration_count": 0,
        "max_iterations": 3,
        "react_agent": good_react_agent
    }
    
    result = good_app.invoke(initial_state)
    
    print(f"Final messages count: {len(result['messages'])}")
    print(f"Iteration count: {result['iteration_count']}")
    print(f"Was approved: {result['is_approved']}")
    print(f"Final output: {result['current_output'][:100]}...")
    print(f"Review feedback: {result['review_feedback']}")
    print()


if __name__ == "__main__":
    test_basic_functionality()
    test_custom_react_agent()
    test_good_custom_agent()
    print("All tests completed!")
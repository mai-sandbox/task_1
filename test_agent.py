#!/usr/bin/env python3
"""
Test script for the dual-agent system.
"""

import os
from langchain_core.messages import HumanMessage
from agent import app, create_dual_agent_graph

def test_basic_functionality():
    """Test basic functionality with a simple question."""
    print("Testing basic functionality...")
    
    initial_state = {
        "messages": [HumanMessage("What's 2 + 2?")],
        "max_iterations": 2
    }
    
    result = app.invoke(initial_state)
    print(f"Final result: {result['messages'][-1].content}")
    print(f"Iterations used: {result.get('iteration_count', 0)}")
    print(f"Review approved: {result.get('review_approved', False)}")
    print("---")

def test_custom_agent():
    """Test with custom configuration."""
    print("Testing custom configuration...")
    
    # Create custom agent with specific review criteria
    custom_review = """
    The response must be mathematically correct and include step-by-step reasoning.
    Respond with 'APPROVED' if these criteria are met, or 'REJECTED: [reason]' if not.
    """
    
    custom_app = create_dual_agent_graph(
        review_criteria=custom_review,
        max_iterations=1
    )
    
    initial_state = {
        "messages": [HumanMessage("Calculate the area of a circle with radius 5")],
    }
    
    result = custom_app.invoke(initial_state)
    print(f"Custom result: {result['messages'][-1].content}")
    print(f"Review result: {result.get('review_result', 'No review')}")
    print("---")

def test_weather_query():
    """Test weather functionality."""
    print("Testing weather query...")
    
    initial_state = {
        "messages": [HumanMessage("What's the weather like in New York?")],
        "max_iterations": 2
    }
    
    result = app.invoke(initial_state)
    print(f"Weather result: {result['messages'][-1].content}")
    print(f"Review approved: {result.get('review_approved', False)}")
    print("---")

if __name__ == "__main__":
    # Only run tests if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping live tests.")
        print("✅ Agent structure is valid and imports successfully.")
    else:
        print("🚀 Running dual-agent system tests...\n")
        
        try:
            test_basic_functionality()
            test_custom_agent() 
            test_weather_query()
            print("✅ All tests completed successfully!")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
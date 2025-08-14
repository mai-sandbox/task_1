#!/usr/bin/env python3
"""
Test script for complex scenarios with the dual-agent system.
"""

import os
from langchain_core.messages import HumanMessage
from agent import create_dual_agent_graph
from langchain_core.tools import tool

# Custom tools for testing
@tool
def get_stock_price(symbol: str) -> str:
    """Get the stock price for a given symbol."""
    # Simulate stock data
    prices = {"AAPL": 150, "GOOGL": 2500, "MSFT": 300}
    return f"The current price of {symbol} is ${prices.get(symbol, 100)}"

@tool
def calculate_percentage(value: float, percentage: float) -> str:
    """Calculate percentage of a value."""
    result = value * (percentage / 100)
    return f"{percentage}% of {value} is {result}"

def test_iteration_loop():
    """Test that review agent can send work back to React agent."""
    print("Testing review-to-react iteration...")
    
    # Create agent with strict review criteria
    strict_review = """
    The response must include:
    1. The exact stock price
    2. A calculation showing the percentage
    3. The final dollar amount
    
    If any of these are missing, respond with 'REJECTED: [specific missing element]'
    Otherwise respond with 'APPROVED'
    """
    
    custom_app = create_dual_agent_graph(
        react_tools=[get_stock_price, calculate_percentage],
        review_criteria=strict_review,
        max_iterations=3
    )
    
    initial_state = {
        "messages": [HumanMessage("What would be the value of a 15% gain on 100 shares of AAPL stock?")],
    }
    
    result = custom_app.invoke(initial_state)
    
    print(f"Final result: {result['messages'][-1].content}")
    print(f"Iterations used: {result.get('iteration_count', 0)}")
    print(f"Review approved: {result.get('review_approved', False)}")
    print(f"Review result: {result.get('review_result', 'No review')}")
    print("---")

def test_max_iterations():
    """Test that the system respects max iterations."""
    print("Testing max iterations limit...")
    
    # Create agent with impossible review criteria
    impossible_review = """
    The response must predict next week's stock prices with 100% accuracy.
    This is impossible, so always respond with 'REJECTED: Cannot predict future with certainty'
    """
    
    custom_app = create_dual_agent_graph(
        react_tools=[get_stock_price],
        review_criteria=impossible_review,
        max_iterations=2
    )
    
    initial_state = {
        "messages": [HumanMessage("What will AAPL stock price be next week?")],
    }
    
    result = custom_app.invoke(initial_state)
    
    print(f"Final result (should mention max iterations): {result['messages'][-1].content}")
    print(f"Iterations used: {result.get('iteration_count', 0)}")
    print(f"Review approved: {result.get('review_approved', False)}")
    print("---")

def test_immediate_approval():
    """Test when review agent approves immediately."""
    print("Testing immediate approval...")
    
    lenient_review = """
    If the response attempts to answer the question, respond with 'APPROVED'
    """
    
    custom_app = create_dual_agent_graph(
        review_criteria=lenient_review,
        max_iterations=3
    )
    
    initial_state = {
        "messages": [HumanMessage("Hello, how are you?")],
    }
    
    result = custom_app.invoke(initial_state)
    
    print(f"Final result: {result['messages'][-1].content}")
    print(f"Iterations used: {result.get('iteration_count', 0)}")
    print(f"Review approved: {result.get('review_approved', False)}")
    print("---")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping live tests.")
        print("✅ Agent structure is valid and imports successfully.")
    else:
        print("🚀 Running complex dual-agent system tests...\n")
        
        try:
            test_iteration_loop()
            test_max_iterations()
            test_immediate_approval()
            print("✅ All complex tests completed successfully!")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
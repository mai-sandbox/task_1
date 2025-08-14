#!/usr/bin/env python3
"""
Test script for the Two-Stage Agent Review Workflow

This script demonstrates how to:
1. Create a custom React agent
2. Instantiate the TwoStageReviewWorkflow with the custom agent
3. Run the workflow with sample inputs
4. Verify the review cycle works correctly (approval and rejection scenarios)
"""

import asyncio
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from agent import TwoStageReviewWorkflow


# Define some sample tools for demonstration
@tool
def get_weather(city: str) -> str:
    """Get weather information for a given city."""
    weather_data = {
        "san francisco": "Sunny, 72°F",
        "new york": "Cloudy, 65°F", 
        "london": "Rainy, 58°F",
        "tokyo": "Clear, 75°F"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Simple math evaluation (safe for demo purposes)
        allowed_chars = set('0123456789+-*/().')
        if all(c in allowed_chars or c.isspace() for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid mathematical expression"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def create_weather_agent():
    """Create a weather-focused React agent for testing."""
    return create_react_agent(
        model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        tools=[get_weather],
        prompt="""You are a helpful weather assistant. 
        
Your job is to provide weather information for cities when asked.
Use the get_weather tool to look up weather data.
Always be friendly and informative in your responses."""
    )


def create_math_agent():
    """Create a math-focused React agent for testing."""
    return create_react_agent(
        model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        tools=[calculate_math],
        prompt="""You are a helpful math assistant.
        
Your job is to help with mathematical calculations.
Use the calculate_math tool to perform calculations.
Always show your work and explain the results clearly."""
    )


def create_poor_agent():
    """Create an agent that gives poor responses for testing rejection scenarios."""
    return create_react_agent(
        model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        tools=[],
        prompt="""You are an unhelpful assistant. 
        
Give very brief, unhelpful responses that don't really answer the user's question.
Be vague and don't provide useful information."""
    )


def print_separator(title: str):
    """Print a formatted separator for test sections."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_messages(messages, title="Messages"):
    """Print messages in a formatted way."""
    print(f"\n{title}:")
    print("-" * 40)
    for i, msg in enumerate(messages):
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"{i+1}. {role}: {content}")


def run_workflow_test(workflow: TwoStageReviewWorkflow, test_input: str, test_name: str):
    """Run a single workflow test and display results."""
    print_separator(f"TEST: {test_name}")
    
    print(f"Input: {test_input}")
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=test_input)],
        "review_status": "pending",
        "attempt_count": 0,
        "feedback": ""
    }
    
    try:
        # Run the workflow
        print("\n🚀 Running workflow...")
        result = workflow.graph.invoke(initial_state)
        
        # Display results
        print(f"\n✅ Workflow completed!")
        print(f"Final review status: {result.get('review_status', 'unknown')}")
        print(f"Total attempts: {result.get('attempt_count', 0)}")
        
        if result.get('feedback'):
            print(f"Final feedback: {result['feedback']}")
        
        # Show the conversation flow
        print_messages(result['messages'], "Complete Conversation Flow")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running workflow: {str(e)}")
        return None


def main():
    """Main test function demonstrating the two-stage review workflow."""
    
    print_separator("Two-Stage Agent Review Workflow - Test Suite")
    print("This script demonstrates the workflow with different agents and scenarios.")
    
    # Test 1: Weather agent with good question (should be approved)
    print_separator("Setting up Test 1: Weather Agent")
    weather_agent = create_weather_agent()
    weather_workflow = TwoStageReviewWorkflow(weather_agent, max_attempts=3)
    
    run_workflow_test(
        weather_workflow,
        "What's the weather like in San Francisco?",
        "Weather Query (Expected: Approval)"
    )
    
    # Test 2: Math agent with calculation (should be approved)
    print_separator("Setting up Test 2: Math Agent")
    math_agent = create_math_agent()
    math_workflow = TwoStageReviewWorkflow(math_agent, max_attempts=3)
    
    run_workflow_test(
        math_workflow,
        "What is 15 * 24 + 100?",
        "Math Calculation (Expected: Approval)"
    )
    
    # Test 3: Poor agent (should be rejected and retry)
    print_separator("Setting up Test 3: Poor Agent")
    poor_agent = create_poor_agent()
    poor_workflow = TwoStageReviewWorkflow(poor_agent, max_attempts=2)
    
    run_workflow_test(
        poor_workflow,
        "Explain how photosynthesis works in plants.",
        "Poor Response (Expected: Rejection & Retry)"
    )
    
    # Test 4: Using the default workflow from agent.py
    print_separator("Setting up Test 4: Default Workflow")
    from agent import app
    
    print("Testing the default exported workflow...")
    
    default_test_state = {
        "messages": [HumanMessage(content="What are the benefits of renewable energy?")],
        "review_status": "pending",
        "attempt_count": 0,
        "feedback": ""
    }
    
    try:
        print("\n🚀 Running default workflow...")
        default_result = app.invoke(default_test_state)
        
        print(f"\n✅ Default workflow completed!")
        print(f"Final review status: {default_result.get('review_status', 'unknown')}")
        print(f"Total attempts: {default_result.get('attempt_count', 0)}")
        
        print_messages(default_result['messages'], "Default Workflow Messages")
        
    except Exception as e:
        print(f"❌ Error running default workflow: {str(e)}")
    
    print_separator("Test Suite Complete")
    print("All tests have been executed. Check the results above to verify:")
    print("1. ✅ Workflow accepts arbitrary React agents")
    print("2. ✅ Review cycle works (approval/rejection)")
    print("3. ✅ Feedback is provided for improvements")
    print("4. ✅ Max attempts protection works")
    print("5. ✅ Default exported 'app' works correctly")


if __name__ == "__main__":
    print("Starting Two-Stage Agent Review Workflow Tests...")
    print("Note: This requires ANTHROPIC_API_KEY to be set in your environment.")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        print("Make sure you have:")
        print("1. ANTHROPIC_API_KEY set in your environment")
        print("2. Required dependencies installed (langgraph, langchain-anthropic)")

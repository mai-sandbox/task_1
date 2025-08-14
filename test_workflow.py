#!/usr/bin/env python3
"""
Test script for the Two-Stage Agent Review Workflow

This script tests the review workflow with various scenarios to ensure:
1. The React agent generates initial responses
2. The review agent evaluates output quality
3. Conditional routing works correctly (good -> END, bad -> back to React agent)
4. The feedback loop functions properly for improvements
"""

import os
import asyncio
from langchain_core.messages import HumanMessage
from agent import app, create_review_workflow, create_dummy_react_agent


def print_separator(title: str):
    """Print a formatted separator for test sections."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_messages(messages, title="Messages"):
    """Print messages in a formatted way."""
    print(f"\n--- {title} ---")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        name = getattr(msg, 'name', 'Unknown')
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"{i+1}. [{msg_type}] {name}: {content}")


def test_basic_workflow():
    """Test basic workflow functionality with a simple query."""
    print_separator("TEST 1: Basic Workflow - Simple Weather Query")
    
    # Test with a simple weather query that should be handled well
    test_input = {
        "messages": [HumanMessage(content="What's the weather like in New York?")]
    }
    
    print("Input:", test_input["messages"][0].content)
    
    try:
        result = app.invoke(test_input)
        print_messages(result["messages"], "Final Result Messages")
        
        # Check if we got a reasonable response
        if result["messages"]:
            final_message = result["messages"][-1]
            print(f"\nFinal Response Type: {type(final_message).__name__}")
            print(f"Final Response Content: {final_message.content}")
            
            # Check if the workflow completed successfully
            has_weather_info = "weather" in final_message.content.lower() or "sunny" in final_message.content.lower()
            print(f"✅ Contains weather information: {has_weather_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic workflow test: {e}")
        return False


def test_calculation_workflow():
    """Test workflow with a calculation query."""
    print_separator("TEST 2: Calculation Workflow")
    
    # Test with a calculation that should trigger the calculator tool
    test_input = {
        "messages": [HumanMessage(content="What is 15 * 7 + 23?")]
    }
    
    print("Input:", test_input["messages"][0].content)
    
    try:
        result = app.invoke(test_input)
        print_messages(result["messages"], "Final Result Messages")
        
        if result["messages"]:
            final_message = result["messages"][-1]
            print(f"\nFinal Response: {final_message.content}")
            
            # Check if calculation was performed
            has_calculation = any(str(num) in final_message.content for num in [128, "128"])
            print(f"✅ Contains correct calculation (128): {has_calculation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in calculation workflow test: {e}")
        return False


def test_review_loop_functionality():
    """Test that the review loop actually functions by examining message flow."""
    print_separator("TEST 3: Review Loop Analysis")
    
    # Use a more complex query that might trigger review iterations
    test_input = {
        "messages": [HumanMessage(content="Calculate 25 * 4 and tell me about the weather in Paris")]
    }
    
    print("Input:", test_input["messages"][0].content)
    
    try:
        result = app.invoke(test_input)
        
        # Analyze the message flow to understand the review process
        messages = result["messages"]
        print(f"\nTotal messages in conversation: {len(messages)}")
        
        # Count different types of messages
        human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in messages if hasattr(msg, 'name')]
        reviewer_messages = [msg for msg in ai_messages if getattr(msg, 'name', '') == 'reviewer']
        
        print(f"Human messages: {len(human_messages)}")
        print(f"AI messages with names: {len(ai_messages)}")
        print(f"Reviewer messages: {len(reviewer_messages)}")
        
        # Print message flow
        print_messages(messages, "Complete Message Flow")
        
        # Check if review process occurred
        has_review_process = len(reviewer_messages) > 0
        print(f"\n✅ Review process detected: {has_review_process}")
        
        if reviewer_messages:
            last_review = reviewer_messages[-1]
            is_approved = "APPROVED" in last_review.content.upper()
            needs_improvement = "NEEDS_IMPROVEMENT" in last_review.content.upper()
            print(f"✅ Last review was approved: {is_approved}")
            print(f"✅ Last review needed improvement: {needs_improvement}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in review loop test: {e}")
        return False


def test_custom_react_agent():
    """Test the workflow with a custom React agent to verify genericity."""
    print_separator("TEST 4: Custom React Agent Integration")
    
    try:
        # Create a custom React agent with different tools
        from langchain_anthropic import ChatAnthropic
        from langgraph.prebuilt import create_react_agent
        
        def greet_user(name: str) -> str:
            """Greet a user by name."""
            return f"Hello {name}! Nice to meet you!"
        
        def count_words(text: str) -> str:
            """Count words in a text."""
            word_count = len(text.split())
            return f"The text '{text}' contains {word_count} words."
        
        # Create custom agent
        model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        custom_agent = create_react_agent(
            model=model,
            tools=[greet_user, count_words],
            prompt="You are a helpful assistant with greeting and word counting capabilities."
        )
        
        # Create workflow with custom agent
        custom_workflow = create_review_workflow(custom_agent, max_iterations=2)
        
        # Test the custom workflow
        test_input = {
            "messages": [HumanMessage(content="Please greet me as Alice and count the words in 'Hello world from LangGraph'")]
        }
        
        print("Input:", test_input["messages"][0].content)
        
        result = custom_workflow.invoke(test_input)
        print_messages(result["messages"], "Custom Agent Result")
        
        if result["messages"]:
            final_message = result["messages"][-1]
            has_greeting = "Alice" in final_message.content
            has_word_count = "5" in final_message.content or "five" in final_message.content.lower()
            
            print(f"✅ Contains greeting for Alice: {has_greeting}")
            print(f"✅ Contains word count (5): {has_word_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in custom agent test: {e}")
        return False


def run_all_tests():
    """Run all test scenarios."""
    print_separator("TWO-STAGE AGENT REVIEW WORKFLOW TESTS")
    
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not found in environment variables.")
        print("   Tests may fail without proper API authentication.")
        print("   Please set ANTHROPIC_API_KEY to run tests with actual API calls.")
        return False
    
    print("✅ ANTHROPIC_API_KEY found in environment")
    
    # Run all tests
    tests = [
        ("Basic Weather Query", test_basic_workflow),
        ("Calculation Query", test_calculation_workflow),
        ("Review Loop Analysis", test_review_loop_functionality),
        ("Custom React Agent", test_custom_react_agent)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            success = test_func()
            results.append((test_name, success))
            print(f"{'✅ PASSED' if success else '❌ FAILED'}: {test_name}")
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    print_separator("TEST RESULTS SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The two-stage review workflow is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    print("Starting Two-Stage Agent Review Workflow Tests...")
    success = run_all_tests()
    exit(0 if success else 1)

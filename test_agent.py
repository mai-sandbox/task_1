#!/usr/bin/env python3
"""
Test script for the Review Agent System

This script demonstrates how the generic review agent system works
with different types of tasks and tools.
"""

from agent import build_review_agent_system, example_calculator, example_text_analyzer
from langchain_core.tools import tool


@tool
def custom_tool(input_text: str) -> str:
    """A custom tool for demonstration"""
    return f"Custom tool processed: {input_text.upper()}"


def test_basic_functionality():
    """Test the basic review agent system"""
    print("=" * 60)
    print("TESTING BASIC REVIEW AGENT SYSTEM")
    print("=" * 60)
    
    # Build the system with default configuration
    app = build_review_agent_system(max_iterations=2)
    
    test_task = "Calculate the sum of 25 + 17 and analyze the result"
    
    initial_state = {
        "messages": [],
        "task": test_task,
        "react_output": None,
        "review_feedback": None,
        "iteration_count": 0,
        "max_iterations": 2,
        "is_approved": False
    }
    
    print(f"Task: {test_task}")
    print("\nExecution Flow:")
    print("-" * 40)
    
    for i, event in enumerate(app.stream(initial_state)):
        for node, data in event.items():
            print(f"\nStep {i+1} - Node: {node}")
            if "messages" in data and data["messages"]:
                for msg in data["messages"]:
                    print(f"  Output: {msg.content}")
            
            # Show key state updates
            if "iteration_count" in data:
                print(f"  Iteration: {data['iteration_count']}")
            if "is_approved" in data:
                print(f"  Approved: {data['is_approved']}")


def test_custom_tools():
    """Test with custom tools"""
    print("\n" + "=" * 60)
    print("TESTING WITH CUSTOM TOOLS")
    print("=" * 60)
    
    custom_tools = [custom_tool, example_calculator]
    app = build_review_agent_system(tools=custom_tools, max_iterations=3)
    
    test_task = "Use the custom tool to process 'hello world' text"
    
    initial_state = {
        "messages": [],
        "task": test_task,
        "react_output": None,
        "review_feedback": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "is_approved": False
    }
    
    print(f"Task: {test_task}")
    print("Available Tools: custom_tool, example_calculator")
    print("\nExecution Flow:")
    print("-" * 40)
    
    for i, event in enumerate(app.stream(initial_state)):
        for node, data in event.items():
            print(f"\nStep {i+1} - Node: {node}")
            if "messages" in data and data["messages"]:
                for msg in data["messages"]:
                    print(f"  Output: {msg.content}")


def test_max_iterations():
    """Test maximum iteration limit"""
    print("\n" + "=" * 60)
    print("TESTING MAXIMUM ITERATION BEHAVIOR")
    print("=" * 60)
    
    app = build_review_agent_system(max_iterations=1)
    
    test_task = "This is a complex task that might require revision"
    
    initial_state = {
        "messages": [],
        "task": test_task,
        "react_output": None,
        "review_feedback": None,
        "iteration_count": 0,
        "max_iterations": 1,
        "is_approved": False
    }
    
    print(f"Task: {test_task}")
    print("Max Iterations: 1")
    print("\nExecution Flow:")
    print("-" * 40)
    
    for i, event in enumerate(app.stream(initial_state)):
        for node, data in event.items():
            print(f"\nStep {i+1} - Node: {node}")
            if "messages" in data and data["messages"]:
                for msg in data["messages"]:
                    print(f"  Output: {msg.content}")


if __name__ == "__main__":
    print("Review Agent System Test Suite")
    print("=" * 60)
    
    test_basic_functionality()
    test_custom_tools()
    test_max_iterations()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
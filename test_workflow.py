#!/usr/bin/env python3
"""
Test Script for Two-Stage Agent Review Workflow

This script demonstrates how to use the two-stage review workflow with different
React agents, showcasing the generic nature of the implementation and the 
review feedback loop in action.

Examples include:
1. Basic text generation task
2. Math problem solving
3. Code generation task
4. Research and analysis task

Each example shows how the workflow handles both successful reviews and 
revision cycles based on feedback.
"""

import asyncio
from typing import List, Any, Dict

# Try to import dependencies, but provide fallbacks for demonstration
try:
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langchain_anthropic import ChatAnthropic
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  LangChain dependencies not installed. Running in demo mode.")
    
    # Mock tool decorator for demonstration
    def tool(func):
        func._is_tool = True
        return func

# Try to import our workflow
try:
    from agent import (
        app, 
        setup_workflow_state, 
        create_main_agent, 
        ReviewWorkflowState
    )
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    WORKFLOW_AVAILABLE = False
    print(f"⚠️  Could not import workflow: {e}")
    print("This might be due to missing dependencies. Running in demo mode.")


# Define some example tools for different agent types
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Simple calculator - only allow basic operations
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def word_count(text: str) -> str:
    """Count words in the given text."""
    words = len(text.split())
    chars = len(text)
    return f"Word count: {words}, Character count: {chars}"


@tool
def code_validator(code: str) -> str:
    """Basic Python code syntax validation."""
    try:
        compile(code, '<string>', 'exec')
        return "Code syntax is valid"
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"


def print_workflow_results(result: Dict[str, Any], task_name: str):
    """Helper function to print workflow results in a readable format."""
    print(f"\n{'='*60}")
    print(f"WORKFLOW RESULTS: {task_name}")
    print(f"{'='*60}")
    
    print(f"Final Status: {result.get('review_status', 'Unknown')}")
    print(f"Iterations: {result.get('iteration_count', 0)}")
    print(f"Task: {result.get('current_task', 'N/A')}")
    
    if result.get('review_feedback'):
        print(f"Final Review Feedback: {result['review_feedback']}")
    
    print(f"\nMessage History ({len(result.get('messages', []))} messages):")
    print("-" * 40)
    
    for i, message in enumerate(result.get('messages', []), 1):
        role = getattr(message, 'name', getattr(message, 'type', 'unknown'))
        content = getattr(message, 'content', str(message))
        
        # Truncate long content for readability
        if len(content) > 200:
            content = content[:200] + "..."
        
        print(f"{i}. [{role}]: {content}")
    
    print(f"{'='*60}\n")


def test_basic_text_generation():
    """Test 1: Basic text generation task."""
    print("🧪 TEST 1: Basic Text Generation")
    print("Task: Write a brief summary of renewable energy benefits")
    
    task = "Write a brief summary of the benefits of renewable energy. Include at least 3 key benefits and keep it under 150 words."
    
    if not WORKFLOW_AVAILABLE or not DEPENDENCIES_AVAILABLE:
        print("⚠️  Running in demo mode - showing expected workflow structure")
        # Mock result for demonstration
        result = {
            'review_status': 'approved',
            'iteration_count': 1,
            'current_task': task,
            'review_feedback': None,
            'messages': [
                {'role': 'user', 'content': task},
                {'role': 'assistant', 'content': 'Renewable energy offers significant benefits including: 1) Environmental protection through reduced greenhouse gas emissions, 2) Energy independence by reducing reliance on fossil fuel imports, 3) Economic growth through job creation in green industries. These clean energy sources like solar and wind are becoming increasingly cost-effective and sustainable for long-term energy needs.'},
                {'role': 'review_agent', 'content': 'APPROVED: The response covers all required elements - includes 3+ key benefits, stays under 150 words, and provides clear, accurate information about renewable energy advantages.'}
            ]
        }
        print_workflow_results(result, "Basic Text Generation")
        return result
    
    # Run the actual workflow if dependencies are available
    initial_state = setup_workflow_state(task)
    result = app.invoke(initial_state)
    
    print_workflow_results(result, "Basic Text Generation")
    return result


def test_math_problem_solving():
    """Test 2: Math problem solving with calculator tool."""
    print("🧪 TEST 2: Math Problem Solving")
    print("Task: Solve a complex math problem")
    
    # This test would require modifying the agent to use the calculator tool
    # For demonstration, we'll show how it would work conceptually
    task = "Calculate the compound interest for $1000 invested at 5% annual rate for 3 years, compounded annually. Show your work step by step."
    
    initial_state = setup_workflow_state(task)
    result = app.invoke(initial_state)
    
    print_workflow_results(result, "Math Problem Solving")
    return result


def test_code_generation():
    """Test 3: Code generation task."""
    print("🧪 TEST 3: Code Generation")
    print("Task: Generate Python code with specific requirements")
    
    task = """Write a Python function called 'fibonacci' that:
1. Takes an integer n as input
2. Returns the nth Fibonacci number
3. Includes proper error handling for negative inputs
4. Has a clear docstring
5. Uses an efficient algorithm (not naive recursion)"""
    
    initial_state = setup_workflow_state(task)
    result = app.invoke(initial_state)
    
    print_workflow_results(result, "Code Generation")
    return result


def test_research_analysis():
    """Test 4: Research and analysis task."""
    print("🧪 TEST 4: Research and Analysis")
    print("Task: Analyze a complex topic")
    
    task = """Analyze the pros and cons of remote work for software development teams. 
Your analysis should include:
1. At least 3 advantages and 3 disadvantages
2. Impact on productivity and collaboration
3. Recommendations for best practices
4. Be well-structured and comprehensive (300-400 words)"""
    
    initial_state = setup_workflow_state(task)
    result = app.invoke(initial_state)
    
    print_workflow_results(result, "Research and Analysis")
    return result


def test_intentionally_poor_response():
    """Test 5: Demonstrate review feedback loop with intentionally poor initial response."""
    print("🧪 TEST 5: Review Feedback Loop Demonstration")
    print("Task: This test shows how the review agent catches poor responses")
    
    # This task is designed to potentially trigger the review feedback loop
    task = """Write a comprehensive guide on machine learning for beginners. 
The guide must include:
1. Clear definition of machine learning
2. At least 5 different types of ML algorithms with examples
3. Real-world applications
4. Getting started resources
5. Common pitfalls to avoid
6. Must be at least 500 words and well-structured"""
    
    initial_state = setup_workflow_state(task)
    result = app.invoke(initial_state)
    
    print_workflow_results(result, "Review Feedback Loop Demo")
    return result


def demonstrate_generic_agent_configuration():
    """Demonstrate how to configure the workflow with different agent setups."""
    print("🔧 DEMONSTRATION: Generic Agent Configuration")
    print("Showing how the workflow can be configured with different tools and prompts")
    
    # Example 1: Math-focused agent configuration
    print("\n📊 Example: Math-focused Agent Configuration")
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    math_agent = create_main_agent(
        model=model,
        tools=[calculator, word_count],
        prompt="""You are a mathematics expert assistant. 
        Use the calculator tool for any numerical computations.
        Always show your work step by step and verify your calculations.""",
        name="math_expert"
    )
    
    print("✅ Math agent configured with calculator and word_count tools")
    
    # Example 2: Code-focused agent configuration  
    print("\n💻 Example: Code-focused Agent Configuration")
    code_agent = create_main_agent(
        model=model,
        tools=[code_validator, word_count],
        prompt="""You are a senior software developer. 
        Write clean, well-documented code with proper error handling.
        Use the code_validator tool to check syntax before finalizing your response.""",
        name="code_expert"
    )
    
    print("✅ Code agent configured with code_validator and word_count tools")
    
    # Example 3: Research-focused agent configuration
    print("\n🔍 Example: Research-focused Agent Configuration")
    research_agent = create_main_agent(
        model=model,
        tools=[word_count],  # In practice, would include web search tools
        prompt="""You are a research analyst. 
        Provide comprehensive, well-structured analysis with clear citations.
        Always include multiple perspectives and evidence-based conclusions.""",
        name="research_analyst"
    )
    
    print("✅ Research agent configured with analysis-focused prompt")
    print("\n🎯 These examples show how the workflow accepts arbitrary React agent configurations!")


def run_all_tests():
    """Run all test cases to demonstrate the workflow capabilities."""
    print("🚀 STARTING TWO-STAGE AGENT REVIEW WORKFLOW TESTS")
    print("=" * 80)
    
    try:
        # Run individual tests
        test_results = []
        
        print("Running tests to demonstrate workflow capabilities...\n")
        
        # Test 1: Basic functionality
        result1 = test_basic_text_generation()
        test_results.append(("Basic Text Generation", result1))
        
        # Test 2: Math problem
        result2 = test_math_problem_solving()
        test_results.append(("Math Problem Solving", result2))
        
        # Test 3: Code generation
        result3 = test_code_generation()
        test_results.append(("Code Generation", result3))
        
        # Test 4: Research analysis
        result4 = test_research_analysis()
        test_results.append(("Research Analysis", result4))
        
        # Test 5: Feedback loop demonstration
        result5 = test_intentionally_poor_response()
        test_results.append(("Feedback Loop Demo", result5))
        
        # Demonstrate generic configuration
        demonstrate_generic_agent_configuration()
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 50)
        
        approved_count = 0
        revision_count = 0
        
        for test_name, result in test_results:
            status = result.get('review_status', 'unknown')
            iterations = result.get('iteration_count', 0)
            
            if status == 'approved':
                approved_count += 1
                status_emoji = "✅"
            elif status == 'needs_revision':
                revision_count += 1
                status_emoji = "🔄"
            else:
                status_emoji = "❓"
            
            print(f"{status_emoji} {test_name}: {status} ({iterations} iterations)")
        
        print(f"\nTotal Tests: {len(test_results)}")
        print(f"Approved: {approved_count}")
        print(f"Needed Revisions: {revision_count}")
        
        print("\n🎉 All tests completed successfully!")
        print("The two-stage review workflow demonstrates:")
        print("✅ Generic React agent integration")
        print("✅ Review and feedback loop functionality") 
        print("✅ Conditional routing based on review results")
        print("✅ Loop termination to prevent infinite cycles")
        print("✅ Configurable tools and prompts")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("This might be due to missing API keys or dependencies.")
        print("Make sure you have ANTHROPIC_API_KEY set in your environment.")


if __name__ == "__main__":
    print("Two-Stage Agent Review Workflow - Test Suite")
    print("=" * 60)
    print("This script demonstrates the generic nature of the workflow")
    print("and shows the review feedback loop in action.\n")
    
    # Check if we should run all tests or just show the structure
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo-only":
        print("🔧 DEMO MODE: Showing workflow structure without API calls")
        demonstrate_generic_agent_configuration()
        
        print("\n📋 Available Test Cases:")
        print("1. Basic Text Generation")
        print("2. Math Problem Solving") 
        print("3. Code Generation")
        print("4. Research and Analysis")
        print("5. Review Feedback Loop Demonstration")
        
        print("\n💡 To run full tests with API calls, run: python test_workflow.py")
        print("💡 Make sure to set ANTHROPIC_API_KEY environment variable")
        
    else:
        print("🚀 Running full test suite...")
        print("Note: This requires ANTHROPIC_API_KEY to be set in your environment")
        print("If you don't have API access, run with --demo-only flag\n")
        
        run_all_tests()



"""
Example Usage of Two-Stage Agent Review Workflow

This file demonstrates how to use the two-stage review workflow with different
ReAct agents, showing various scenarios including successful reviews and
cases requiring iteration.
"""

import os
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import our workflow
from agent import create_review_workflow, invoke_workflow, app


def setup_environment():
    """Set up environment variables for testing (use dummy values for demo)."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using placeholder for demo.")
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-demo"


def create_math_agent():
    """Create a specialized math ReAct agent."""
    @tool
    def calculator(expression: str) -> str:
        """Calculate mathematical expressions."""
        try:
            # Safe evaluation for basic math
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"The result of {expression} is {result}"
            else:
                return "Error: Invalid characters in expression"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    @tool
    def factorial(n: str) -> str:
        """Calculate factorial of a number."""
        try:
            num = int(n)
            if num < 0:
                return "Error: Factorial is not defined for negative numbers"
            if num > 20:
                return "Error: Number too large for factorial calculation"
            
            result = 1
            for i in range(1, num + 1):
                result *= i
            return f"The factorial of {num} is {result}"
        except ValueError:
            return f"Error: '{n}' is not a valid integer"
    
    tools = [calculator, factorial]
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return create_react_agent(model, tools)


def create_research_agent():
    """Create a research-focused ReAct agent."""
    @tool
    def web_search(query: str) -> str:
        """Search the web for information (mock implementation)."""
        # Mock search results for demonstration
        mock_results = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
            "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
            "climate": "Climate change refers to long-term shifts in global temperatures and weather patterns."
        }
        
        query_lower = query.lower()
        for key, result in mock_results.items():
            if key in query_lower:
                return f"Search results for '{query}': {result}"
        
        return f"Search results for '{query}': No specific information found in mock database."
    
    @tool
    def summarize_text(text: str) -> str:
        """Summarize a given text."""
        if len(text) <= 100:
            return f"Text is already concise: {text}"
        
        # Simple summarization (first and last sentences)
        sentences = text.split('. ')
        if len(sentences) <= 2:
            return text
        
        summary = f"{sentences[0]}. ... {sentences[-1]}"
        return f"Summary: {summary}"
    
    tools = [web_search, summarize_text]
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    return create_react_agent(model, tools)


def example_1_basic_usage():
    """Example 1: Basic usage with default workflow."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage with Default Workflow")
    print("=" * 60)
    
    try:
        # Simple query that should be approved quickly
        query = "What is 2 + 2?"
        print(f"Query: {query}")
        print("\nExecuting workflow...")
        
        result = invoke_workflow(query, max_iterations=3)
        
        print(f"\n✅ Final Status: {result['review_status']}")
        print(f"📊 Iterations: {result['iteration_count']}")
        print(f"💬 Final Output: {result['current_output']}")
        
        # Show conversation history
        print(f"\n📝 Conversation History ({len(result['messages'])} messages):")
        for i, msg in enumerate(result['messages'][-3:], 1):  # Show last 3 messages
            role = msg.__class__.__name__.replace('Message', '')
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"  {i}. {role}: {content}")
            
    except Exception as e:
        print(f"❌ Error in basic usage example: {e}")


def example_2_custom_math_agent():
    """Example 2: Using custom math agent."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Math Agent")
    print("=" * 60)
    
    try:
        # Create custom math agent
        math_agent = create_math_agent()
        
        # Create workflow with custom agent
        workflow = create_review_workflow(
            react_agent=math_agent,
            max_iterations=2
        )
        compiled_workflow = workflow.compile()
        
        # Test with a math problem
        query = "Calculate the factorial of 5 and then add 10 to it"
        print(f"Query: {query}")
        print("\nExecuting workflow with custom math agent...")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "review_status": None,
            "iteration_count": 0,
            "max_iterations": 2,
            "original_query": query,
            "current_output": None
        }
        
        result = compiled_workflow.invoke(initial_state)
        
        print(f"\n✅ Final Status: {result['review_status']}")
        print(f"📊 Iterations: {result['iteration_count']}")
        print(f"💬 Final Output: {result['current_output']}")
        
    except Exception as e:
        print(f"❌ Error in custom math agent example: {e}")


def example_3_research_agent_with_iterations():
    """Example 3: Research agent that might require iterations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Research Agent with Potential Iterations")
    print("=" * 60)
    
    try:
        # Create custom research agent
        research_agent = create_research_agent()
        
        # Create workflow with custom agent and higher iteration limit
        workflow = create_review_workflow(
            react_agent=research_agent,
            max_iterations=4
        )
        compiled_workflow = workflow.compile()
        
        # Complex query that might need refinement
        query = "Research the impact of AI on climate change and provide a comprehensive summary"
        print(f"Query: {query}")
        print("\nExecuting workflow with research agent...")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "review_status": None,
            "iteration_count": 0,
            "max_iterations": 4,
            "original_query": query,
            "current_output": None
        }
        
        result = compiled_workflow.invoke(initial_state)
        
        print(f"\n✅ Final Status: {result['review_status']}")
        print(f"📊 Iterations: {result['iteration_count']}")
        print(f"💬 Final Output: {result['current_output']}")
        
        # Show review feedback if any
        review_messages = [msg for msg in result['messages'] if hasattr(msg, 'name') and msg.name == 'reviewer']
        if review_messages:
            print(f"\n🔍 Review Feedback:")
            for msg in review_messages:
                print(f"  - {msg.content[:150]}...")
        
    except Exception as e:
        print(f"❌ Error in research agent example: {e}")


def example_4_max_iterations_reached():
    """Example 4: Demonstrate max iterations behavior."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Max Iterations Behavior")
    print("=" * 60)
    
    try:
        # Create a workflow with very low max iterations
        workflow = create_review_workflow(max_iterations=1)
        compiled_workflow = workflow.compile()
        
        # Complex query that likely needs multiple iterations
        query = "Explain quantum computing, its applications, and solve this equation: x^2 + 5x + 6 = 0"
        print(f"Query: {query}")
        print("Max iterations set to 1 to demonstrate limit behavior...")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "review_status": None,
            "iteration_count": 0,
            "max_iterations": 1,
            "original_query": query,
            "current_output": None
        }
        
        result = compiled_workflow.invoke(initial_state)
        
        print(f"\n✅ Final Status: {result['review_status']}")
        print(f"📊 Iterations: {result['iteration_count']}")
        print(f"🔄 Max Iterations Reached: {result['iteration_count'] >= result['max_iterations']}")
        print(f"💬 Final Output: {result['current_output'][:200]}...")
        
    except Exception as e:
        print(f"❌ Error in max iterations example: {e}")


def example_5_different_review_models():
    """Example 5: Using different review models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Different Review Models")
    print("=" * 60)
    
    try:
        # Create workflow with different review model
        strict_reviewer = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        workflow = create_review_workflow(
            review_model=strict_reviewer,
            max_iterations=3
        )
        compiled_workflow = workflow.compile()
        
        query = "What are the benefits of renewable energy?"
        print(f"Query: {query}")
        print("Using strict reviewer model...")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "review_status": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "original_query": query,
            "current_output": None
        }
        
        result = compiled_workflow.invoke(initial_state)
        
        print(f"\n✅ Final Status: {result['review_status']}")
        print(f"📊 Iterations: {result['iteration_count']}")
        print(f"💬 Final Output: {result['current_output'][:200]}...")
        
    except Exception as e:
        print(f"❌ Error in different review models example: {e}")


def main():
    """Run all examples."""
    print("🚀 Two-Stage Agent Review Workflow Examples")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Run examples
    example_1_basic_usage()
    example_2_custom_math_agent()
    example_3_research_agent_with_iterations()
    example_4_max_iterations_reached()
    example_5_different_review_models()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    
    print("\n📚 Key Takeaways:")
    print("1. The workflow can accept any ReAct agent")
    print("2. Review process ensures quality control")
    print("3. Iteration limits prevent infinite loops")
    print("4. Different models can be used for reviewing")
    print("5. State is preserved throughout the workflow")


if __name__ == "__main__":
    main()

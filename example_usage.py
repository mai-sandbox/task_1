#!/usr/bin/env python3
"""
Example usage of the orchestrator agent with different ReAct agents
"""

from langchain_core.messages import HumanMessage, AIMessage
from agent import app, create_app_with_custom_agent


def example_custom_react_agent(state):
    """
    Example of how to create a custom ReAct agent.
    
    This is a simple example - in practice, you would replace this with
    your sophisticated ReAct agent that does reasoning, action-taking, etc.
    """
    messages = state.get("messages", [])
    
    # Get the last human message
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break
    
    if not last_human_message:
        response = "No input provided"
    else:
        user_input = last_human_message.content
        
        # Simulate ReAct-style reasoning
        response = f"""
        Let me think through this step by step:
        
        Thought: The user is asking about '{user_input}'. I need to analyze this carefully.
        
        Action: I'll consider the key aspects and provide a comprehensive response.
        
        Observation: Based on my analysis, I can provide valuable insights.
        
        Thought: I now have enough information to provide a complete answer.
        
        Final Answer: Here's my thorough response to '{user_input}': This is a well-reasoned and comprehensive answer that addresses all aspects of your question with detailed explanations and relevant examples.
        """
    
    return {
        "messages": messages + [AIMessage(content=response.strip())]
    }


def run_example():
    """Run example usage of the orchestrator agent"""
    
    print("🤖 Orchestrator Agent Example\n")
    
    # Example 1: Using the default agent
    print("=" * 50)
    print("Example 1: Using the default ReAct agent")
    print("=" * 50)
    
    result = app.invoke({
        "messages": [HumanMessage("Explain the benefits of renewable energy")],
        "react_output": None,
        "review_result": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "react_agent": None
    })
    
    print(f"Iterations used: {result['iteration_count']}")
    print(f"Final response:\n{result['messages'][-1].content}")
    
    # Example 2: Using a custom ReAct agent
    print("\n" + "=" * 50)
    print("Example 2: Using a custom ReAct agent")
    print("=" * 50)
    
    custom_app = create_app_with_custom_agent(example_custom_react_agent, max_iterations=2)
    
    result2 = custom_app.invoke({
        "messages": [HumanMessage("What are the key principles of machine learning?")],
        "react_output": None,
        "review_result": None,
        "iteration_count": 0,
        "max_iterations": 2,
        "react_agent": None
    })
    
    print(f"Iterations used: {result2['iteration_count']}")
    print(f"Final response:\n{result2['messages'][-1].content}")
    
    # Example 3: Show what happens with a failing agent
    print("\n" + "=" * 50)
    print("Example 3: Testing with an agent that needs multiple iterations")
    print("=" * 50)
    
    def inconsistent_agent(state):
        """An agent that sometimes gives good and sometimes bad responses"""
        import random
        messages = state.get("messages", [])
        
        # Sometimes give a short (bad) response, sometimes a good one
        if random.random() < 0.5:  # 50% chance of short response
            response = "Short answer"
        else:
            response = "This is a comprehensive and well-thought-out response that provides detailed information and analysis."
        
        return {
            "messages": messages + [AIMessage(content=response)]
        }
    
    inconsistent_app = create_app_with_custom_agent(inconsistent_agent, max_iterations=3)
    
    result3 = inconsistent_app.invoke({
        "messages": [HumanMessage("Tell me about artificial intelligence")],
        "react_output": None,
        "review_result": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "react_agent": None
    })
    
    print(f"Iterations used: {result3['iteration_count']}")
    print(f"Final response:\n{result3['messages'][-1].content}")


if __name__ == "__main__":
    run_example()
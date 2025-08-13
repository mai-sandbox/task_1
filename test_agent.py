#!/usr/bin/env python3

from langchain_core.messages import HumanMessage
from agent import app


def test_review_loop_agent():
    """Test the review loop agent with a simple question."""
    
    initial_state = {
        "messages": [HumanMessage("What is the capital of France?")]
    }
    
    print("Testing Review Loop Agent")
    print("=" * 40)
    print(f"Initial question: {initial_state['messages'][0].content}")
    print()
    
    try:
        # Run the agent
        result = app.invoke(initial_state)
        
        print("Final Result:")
        print("-" * 20)
        
        # Print all messages
        for i, message in enumerate(result["messages"]):
            role = "Human" if hasattr(message, 'type') and message.type == "human" else "AI"
            print(f"{i+1}. {role}: {message.content}")
            print()
        
        # Print additional state information
        print("Additional State:")
        print(f"- Iteration Count: {result.get('iteration_count', 'N/A')}")
        print(f"- Is Approved: {result.get('is_approved', 'N/A')}")
        print(f"- Final Review Feedback: {result.get('review_feedback', 'N/A')}")
        
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_review_loop_agent()
"""
Complex test demonstrating the feedback loop functionality
"""
from agent import create_custom_dual_agent
from langchain_core.messages import HumanMessage


def test_feedback_loop():
    """Test that demonstrates the feedback loop in action"""
    print("=== Testing Feedback Loop with Improving Agent ===")
    
    iteration_responses = [
        "No.",  # First attempt - too short
        "I don't know anything about that topic.",  # Second attempt - contains failure indicator
        "I'll help you understand programming concepts. Programming is the process of creating instructions for computers to follow. It involves writing code using various programming languages to solve problems and create applications."  # Third attempt - should pass
    ]
    
    response_index = [0]  # Using list to make it mutable in closure
    
    def improving_react_agent(messages):
        """React agent that improves its responses based on feedback"""
        current_response = iteration_responses[response_index[0] % len(iteration_responses)]
        response_index[0] += 1
        
        # If there's system feedback in messages, show that we're responding to it
        has_feedback = any(msg.type == "system" and "feedback" in msg.content.lower() for msg in messages if hasattr(msg, 'type'))
        if has_feedback and response_index[0] > 1:
            print(f"  Agent received feedback, generating attempt #{response_index[0]-1}")
        
        return current_response
    
    improving_app = create_custom_dual_agent(improving_react_agent, max_iterations=4)
    
    initial_state = {
        "messages": [HumanMessage("Can you explain what programming is?")],
        "current_output": None,
        "review_feedback": None,
        "is_approved": False,
        "iteration_count": 0,
        "max_iterations": 4,
        "react_agent": improving_react_agent
    }
    
    result = improving_app.invoke(initial_state)
    
    print(f"Final iteration count: {result['iteration_count']}")
    print(f"Was finally approved: {result['is_approved']}")
    print(f"Final output: {result['current_output']}")
    print(f"Final review feedback: {result['review_feedback']}")
    print(f"Total messages in conversation: {len(result['messages'])}")
    
    # Print all messages to show the conversation flow
    print("\n=== Full Conversation Flow ===")
    for i, msg in enumerate(result['messages']):
        msg_type = msg.__class__.__name__
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i+1}. {msg_type}: {content}")


if __name__ == "__main__":
    test_feedback_loop()
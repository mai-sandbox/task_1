#!/usr/bin/env python3
"""
Example usage of the review loop agent with a proper ReAct agent.
"""

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from agent import ReviewLoopAgent, create_review_loop_app


def create_better_react_agent():
    """
    Create a better ReAct agent that can actually answer questions.
    This simulates a proper ReAct agent with tool calling capabilities.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    def react_agent(state_input):
        """A ReAct agent that can actually answer questions."""
        messages = state_input.get("messages", [])
        
        if not messages:
            return {"messages": [AIMessage(content="I need a question or request to help you.")]}
        
        # Get the latest human message
        last_human_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_message = msg
                break
        
        if not last_human_message:
            return {"messages": messages + [AIMessage(content="I couldn't find a question to answer.")]}
        
        # Check if this is feedback from the review loop
        is_feedback_message = "Previous review feedback:" in last_human_message.content
        
        if is_feedback_message:
            # Extract the original question from the first message
            original_question = messages[0].content if messages else "Unknown question"
            
            # Create a more focused response incorporating the feedback
            system_prompt = f"""
            You are a helpful AI assistant. The user asked: "{original_question}"
            
            You received this feedback on your previous response: {last_human_message.content}
            
            Please provide a clear, direct, and helpful answer to the original question, taking the feedback into account.
            """
            
            response = llm.invoke([HumanMessage(content=system_prompt)])
        else:
            # First attempt - answer the question directly
            system_prompt = f"""
            You are a helpful AI assistant. Please provide a clear, direct, and informative answer to the following question: {last_human_message.content}
            """
            
            response = llm.invoke([HumanMessage(content=system_prompt)])
        
        # Add the response to messages
        new_messages = messages + [AIMessage(content=response.content)]
        return {"messages": new_messages}
    
    return react_agent


def test_with_better_agent():
    """Test the review loop with a better ReAct agent."""
    
    print("Testing Review Loop Agent with Better ReAct Agent")
    print("=" * 55)
    
    # Create a better ReAct agent
    better_agent = create_better_react_agent()
    
    # Create review loop app with the better agent
    app = create_review_loop_app(
        work_agent=better_agent,
        max_iterations=2  # Reduce to 2 for cleaner output
    )
    
    # Test questions
    test_questions = [
        "What is the capital of France?",
        "Explain what quantum computing is in simple terms.",
        "What are the benefits of renewable energy?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}: {question} ---")
        print()
        
        initial_state = {
            "messages": [HumanMessage(question)]
        }
        
        try:
            result = app.invoke(initial_state)
            
            # Find the final AI response
            final_response = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_response = msg.content
                    break
            
            print(f"Question: {question}")
            print(f"Final Answer: {final_response}")
            print(f"Iterations: {result.get('iteration_count', 'N/A')}")
            print(f"Approved: {result.get('is_approved', 'N/A')}")
            if result.get('review_feedback'):
                print(f"Review Feedback: {result['review_feedback']}")
            
        except Exception as e:
            print(f"Error with question '{question}': {e}")
            import traceback
            traceback.print_exc()


def test_custom_agent_integration():
    """Demonstrate how to integrate your own custom agent."""
    
    print("\n" + "=" * 60)
    print("DEMONSTRATING CUSTOM AGENT INTEGRATION")
    print("=" * 60)
    
    def my_custom_agent(state_input):
        """
        Example of how you would integrate your own custom agent.
        
        Your agent should:
        1. Take a dict with 'messages' key containing a list of BaseMessage objects
        2. Return a dict with updated 'messages' list
        """
        messages = state_input.get("messages", [])
        
        # Your custom logic here
        # This could be your existing ReAct agent, CrewAI agent, etc.
        
        # For this example, we'll simulate a specialized math agent
        last_question = messages[-1].content if messages else ""
        
        if any(word in last_question.lower() for word in ['math', 'calculate', 'solve', '+', '-', '*', '/']):
            response = "I'm a specialized math agent. I can help you with mathematical problems and calculations."
        else:
            response = "I'm designed primarily for math problems, but I'll try to help with your general question."
        
        new_messages = messages + [AIMessage(content=response)]
        return {"messages": new_messages}
    
    # Use your custom agent in the review loop
    app = create_review_loop_app(work_agent=my_custom_agent)
    
    result = app.invoke({
        "messages": [HumanMessage("Help me solve 15 * 24")]
    })
    
    print("Custom Agent Integration Result:")
    for i, msg in enumerate(result["messages"]):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"{i+1}. {role}: {msg.content}")


if __name__ == "__main__":
    test_with_better_agent()
    test_custom_agent_integration()
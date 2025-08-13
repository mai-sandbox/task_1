from agent import app
from langchain_core.messages import HumanMessage

def test_review_loop():
    """Test the review loop agent with a simple query."""
    
    test_cases = [
        "What is 2 + 2?",
        "Explain quantum computing in simple terms",
        "Write a haiku about programming"
    ]
    
    for test_query in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing with query: {test_query}")
        print('='*50)
        
        initial_state = {
            "messages": [HumanMessage(content=test_query)]
        }
        
        try:
            result = app.invoke(initial_state)
            
            print(f"Iterations: {result.get('iteration_count', 0)}")
            print(f"Approved: {result.get('is_approved', False)}")
            print(f"Final Response: {result.get('initial_response', 'No response')}")
            
            if result.get('review_feedback'):
                print(f"Last Feedback: {result.get('review_feedback')}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_review_loop()
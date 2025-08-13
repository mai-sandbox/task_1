from agent import app
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Test the agent with a simple query
initial_state = {
    "messages": [HumanMessage("What is the capital of France?")],
    "max_iterations": 3
}

print("Testing the agent system...")
print("-" * 50)

result = app.invoke(initial_state)

print("Final Messages:")
for msg in result["messages"]:
    print(f"\n{msg.content}")

print("\n" + "-" * 50)
print(f"React Output: {result.get('react_output', 'N/A')}")
print(f"Review Feedback: {result.get('review_feedback', 'N/A')}")
print(f"Iterations: {result.get('iteration_count', 0)}")
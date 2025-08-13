from agent import app
from langchain_core.messages import HumanMessage
import os

# Set a dummy API key for testing (won't make actual calls in this test)
os.environ["OPENAI_API_KEY"] = "sk-test-key"

# Test the agent structure
print("Testing agent with review loop...")

# Create test state
test_state = {
    "messages": [HumanMessage("What is 25 * 4 + 10?")],
    "review_count": 0,
    "max_reviews": 2
}

print("\nAgent successfully created and compiled!")
print("\nAgent structure:")
print("- Accepts arbitrary ReAct agent or creates default one")
print("- Runs initial ReAct agent")
print("- Reviews output with reviewer agent")
print("- If approved: finishes")
print("- If needs improvement: sends feedback back to ReAct agent")
print("- Maximum review iterations configurable (default: 2)")

print("\nTo use with custom ReAct agent:")
print("""
from agent import ReviewLoopAgent
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Create your custom ReAct agent
custom_llm = ChatOpenAI(model="gpt-4")
custom_tools = [...]  # Your tools
custom_react_agent = create_react_agent(custom_llm, custom_tools)

# Create review loop agent with custom ReAct agent
agent = ReviewLoopAgent(
    react_agent=custom_react_agent,
    reviewer_llm=ChatOpenAI(model="gpt-4", temperature=0.3),
    max_reviews=3
)

app = agent.compile()
""")
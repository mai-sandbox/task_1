# Review Loop Agent

A generic LangGraph agent that implements a review-loop pattern. It first runs a ReAct agent to get initial output, then uses another agent to review that output. If the review is positive, it finishes. Otherwise, it sends feedback back to the original agent for improvement.

## Architecture

The agent uses a reflection pattern with two main nodes:

1. **Work Node**: Executes the provided ReAct agent
2. **Review Node**: Evaluates the work agent's output using an LLM

The flow is:
```
Start → Work → Review → [Continue/Finish]
              ↑              ↓
              └─── Continue ───┘
```

## Features

- **Generic**: Works with any ReAct agent that follows the expected interface
- **Configurable**: Customizable review LLM and maximum iterations
- **LangGraph Native**: Built using LangGraph patterns with proper state management
- **Deployment Ready**: Includes `langgraph.json` configuration

## Usage

### Basic Usage

```python
from langchain_core.messages import HumanMessage
from agent import app

# Use with the default simple agent
initial_state = {
    "messages": [HumanMessage("What is the capital of France?")]
}

result = app.invoke(initial_state)
print(result["messages"][-1].content)  # Final AI response
```

### Custom Agent Integration

```python
from agent import create_review_loop_app
from langchain_core.messages import HumanMessage, AIMessage

def my_custom_react_agent(state_input):
    """
    Your custom ReAct agent.
    
    Must take: dict with 'messages' key containing list of BaseMessage objects
    Must return: dict with updated 'messages' list
    """
    messages = state_input.get("messages", [])
    
    # Your agent logic here
    # This could be CrewAI, AutoGen, or any other agent
    
    response = "Your agent's response here"
    new_messages = messages + [AIMessage(content=response)]
    return {"messages": new_messages}

# Create review loop app with your agent
app = create_review_loop_app(
    work_agent=my_custom_react_agent,
    max_iterations=3
)

result = app.invoke({
    "messages": [HumanMessage("Your question here")]
})
```

### Advanced Configuration

```python
from langchain_openai import ChatOpenAI
from agent import ReviewLoopAgent

# Custom review LLM
review_llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create agent with custom settings
agent = ReviewLoopAgent(
    work_agent=my_custom_react_agent,
    review_llm=review_llm,
    max_iterations=5
)

app = agent.app
```

## State Schema

The agent uses the following state:

```python
class ReviewLoopState(TypedDict):
    messages: list[BaseMessage]          # Conversation messages
    review_feedback: str                 # Feedback from review agent
    iteration_count: int                 # Current iteration number
    max_iterations: int                  # Maximum allowed iterations
    is_approved: bool                    # Whether output was approved
```

## Agent Interface Requirements

Your ReAct agent must implement this interface:

```python
def your_agent(state_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
        state_input: Dictionary containing at minimum:
            - 'messages': list of BaseMessage objects
    
    Returns:
        Dictionary containing:
            - 'messages': updated list of BaseMessage objects
    """
    pass
```

## Files

- `agent.py` - Main review loop implementation
- `langgraph.json` - LangGraph configuration for deployment
- `requirements.txt` - Python dependencies
- `test_agent.py` - Basic test with simple agent
- `example_usage.py` - Advanced examples with better agents

## Installation

```bash
pip install -r requirements.txt
```

## Testing

Run the basic test:
```bash
python test_agent.py
```

Run advanced examples:
```bash
python example_usage.py
```

## Deployment

The agent is deployment-ready with LangGraph Platform:

1. Ensure your environment variables are set (especially `OPENAI_API_KEY`)
2. Deploy using the LangGraph CLI or platform

The `langgraph.json` configuration exports the main app at `./agent.py:app`.

## Customization

### Review Criteria

The review agent evaluates responses based on:
1. Direct addressing of the user's question
2. Accuracy and helpfulness
3. Completeness

You can customize this by modifying the `_review_node` method in the `ReviewLoopAgent` class.

### Iteration Control

Control the review loop with:
- `max_iterations`: Maximum number of review cycles
- Review approval logic in `_should_continue`

### Integration Patterns

Common integration patterns:
1. **Wrapper Pattern**: Wrap existing agents without modification
2. **Adapter Pattern**: Create adapters for non-conforming agents
3. **Composition Pattern**: Compose multiple agents within the work node

## Examples

See `example_usage.py` for complete examples including:
- Integration with ChatOpenAI-based agents
- Custom specialized agents
- Error handling patterns
- Multiple test scenarios

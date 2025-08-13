# Dual-Agent System with React Agent and Reviewer

A generic LangGraph implementation that wraps any React agent with an intelligent reviewer to ensure high-quality outputs through iterative improvement.

## Overview

This system implements a dual-agent workflow:
1. **React Agent** generates initial output using tools and reasoning
2. **Reviewer Agent** evaluates the output quality 
3. **Decision Logic** either accepts the output or sends it back for improvement
4. **Iterative Loop** continues until output is approved or max iterations reached

## Key Features

- **Generic Design**: Works with any React agent
- **Quality Assurance**: Automatic review and improvement cycles
- **Configurable**: Adjustable retry limits and review criteria
- **LangGraph Native**: Full integration with LangGraph ecosystem
- **Deployment Ready**: Includes proper configuration files

## Quick Start

### Basic Usage

```python
from langchain_core.messages import HumanMessage
from agent import app

# Use the default system
result = app.invoke({
    "messages": [HumanMessage("Your question here")]
})

print(result["messages"][-1].content)
```

### Custom React Agent Integration

```python
from agent import create_app
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Create your custom React agent
@tool
def my_tool(input: str) -> str:
    return f"Tool result for: {input}"

llm = ChatOpenAI(model="gpt-4o-mini")
my_react_agent = create_react_agent(llm, [my_tool])

# Wrap it with the dual-agent system
enhanced_app = create_app(
    react_agent=my_react_agent,
    max_iterations=3
)

# Use it
result = enhanced_app.invoke({
    "messages": [HumanMessage("Your question")]
})
```

## File Structure

```
./agent.py          # Main agent implementation
./langgraph.json    # LangGraph configuration
./test_agent.py     # Basic functionality test
./test_complex.py   # Complex scenarios test
./example_usage.py  # Usage examples
```

## How It Works

1. **Input Processing**: Receives messages in standard LangGraph format
2. **React Agent Execution**: Runs the provided React agent to generate initial output
3. **Quality Review**: Reviewer agent evaluates the output against criteria:
   - Direct answer to the question
   - Accuracy and reasoning quality
   - Completeness of information
   - Clarity and understanding
4. **Decision Making**: 
   - If approved: Finalizes and returns the result
   - If rejected: Provides feedback and retries (up to max_iterations)
5. **Output**: Returns the final approved response

## State Schema

```python
class State(TypedDict):
    messages: List[BaseMessage]          # Conversation messages
    react_output: Optional[str]          # React agent output
    review_feedback: Optional[str]       # Reviewer feedback
    iteration_count: int                 # Current iteration number
    max_iterations: int                  # Maximum allowed iterations
    is_approved: bool                    # Approval status
    react_agent_graph: Optional[object]  # React agent reference
```

## Configuration

The system is configured via `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:app"
  },
  "env": ".env"
}
```

## Environment Setup

Required environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key
```

## Installation

```bash
pip install langgraph langchain-openai langchain-core
```

## Testing

Run the test suite:
```bash
python test_agent.py          # Basic functionality
python test_complex.py        # Complex scenarios  
python example_usage.py       # Usage examples
```

## Examples

### Research Agent
```python
# Agent with web search and summarization tools
research_agent = create_research_agent()
research_app = create_app(research_agent)
```

### Math Agent  
```python
# Agent with calculation and unit conversion tools
math_agent = create_math_agent()
math_app = create_app(math_agent)
```

### Custom Tools Agent
```python
@tool
def custom_tool(query: str) -> str:
    return f"Custom result for {query}"

custom_agent = create_react_agent(llm, [custom_tool])
custom_app = create_app(custom_agent)
```

## Deployment

This agent is deployment-ready for:
- LangGraph Platform
- Self-hosted LangGraph Server
- Local development server

Deploy using the LangGraph CLI:
```bash
langgraph dev  # Local development
langgraph deploy  # Platform deployment
```

## Benefits

- **Higher Quality**: Automatic review ensures better outputs
- **Reliability**: Iterative improvement reduces errors
- **Flexibility**: Works with any React agent architecture
- **Transparency**: Clear feedback on why outputs were accepted/rejected
- **Scalability**: Built on LangGraph's robust infrastructure

## Use Cases

- Customer support chatbots requiring high accuracy
- Research assistants needing comprehensive answers
- Educational tools requiring clear explanations
- Technical documentation systems
- Any application where output quality is critical

## Architecture

```
Input → React Agent → Reviewer → Decision
  ↑                              ↓
  └── Feedback Loop ←── Retry ←──┘
```

The system maintains conversation context and provides detailed feedback for continuous improvement, making it ideal for applications requiring high-quality, reliable responses.

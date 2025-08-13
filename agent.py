from typing import TypedDict, Annotated, Sequence, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import operator
import os


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    review_count: int
    max_reviews: int


class ReviewLoopAgent:
    def __init__(
        self,
        react_agent=None,
        reviewer_llm=None,
        max_reviews: int = 2,
        tools: list = None
    ):
        self.max_reviews = max_reviews
        
        # Default LLMs if not provided
        if react_agent is None:
            if tools is None:
                # Create a simple calculator tool as default
                @tool
                def calculate(expression: str) -> str:
                    """Evaluate a mathematical expression."""
                    try:
                        result = eval(expression, {"__builtins__": {}})
                        return f"Result: {result}"
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                tools = [calculate]
            
            # Create default ReAct agent
            react_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            self.react_agent = create_react_agent(react_llm, tools)
        else:
            self.react_agent = react_agent
        
        # Default reviewer LLM if not provided
        self.reviewer_llm = reviewer_llm or ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("react_agent", self._run_react_agent)
        workflow.add_node("reviewer", self._review_output)
        workflow.add_node("prepare_retry", self._prepare_retry)
        
        # Set entry point
        workflow.set_entry_point("react_agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "react_agent",
            lambda x: "reviewer",
            {"reviewer": "reviewer"}
        )
        
        workflow.add_conditional_edges(
            "reviewer",
            self._should_retry,
            {
                "retry": "prepare_retry",
                "finish": END
            }
        )
        
        workflow.add_edge("prepare_retry", "react_agent")
        
        return workflow
    
    def _run_react_agent(self, state: AgentState) -> dict:
        """Run the ReAct agent with the current messages."""
        # Extract only the latest relevant message for the react agent
        messages_for_agent = []
        
        # Get the original human message or the latest retry message
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                # Check if this is a retry message with feedback
                if "Previous attempt" in msg.content:
                    messages_for_agent = [msg]
                    break
                else:
                    messages_for_agent = [msg]
        
        # Run the react agent
        result = self.react_agent.invoke({
            "messages": messages_for_agent
        })
        
        # Extract the final message from the react agent
        agent_response = result["messages"][-1]
        
        return {
            "messages": [agent_response],
            "review_count": state.get("review_count", 0) + 1
        }
    
    def _review_output(self, state: AgentState) -> dict:
        """Review the output from the ReAct agent."""
        # Get the latest AI message (from react agent)
        last_message = state["messages"][-1]
        
        # Get the original human request
        original_request = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage) and "Previous attempt" not in msg.content:
                original_request = msg.content
                break
        
        review_prompt = f"""Review the following response to determine if it adequately addresses the user's request.

User's request: {original_request}

Agent's response: {last_message.content}

Evaluate whether the response:
1. Directly addresses the user's question or request
2. Provides accurate and complete information
3. Is clear and well-structured

Respond with ONLY one of these two options:
- "APPROVED" if the response is satisfactory
- "NEEDS_IMPROVEMENT: [specific feedback on what needs to be fixed]" if improvements are needed

Your response:"""
        
        review_result = self.reviewer_llm.invoke([HumanMessage(content=review_prompt)])
        
        # Add review result to messages for tracking
        return {
            "messages": [AIMessage(content=f"[Review]: {review_result.content}")]
        }
    
    def _should_retry(self, state: AgentState) -> str:
        """Determine whether to retry based on review and count."""
        # Check review result
        last_review = state["messages"][-1].content
        
        # Check if we've exceeded max reviews
        if state.get("review_count", 0) >= state.get("max_reviews", self.max_reviews):
            return "finish"
        
        # Check if approved
        if "APPROVED" in last_review:
            return "finish"
        
        return "retry"
    
    def _prepare_retry(self, state: AgentState) -> dict:
        """Prepare state for retry with feedback."""
        # Extract feedback from review
        last_review = state["messages"][-1].content
        feedback = last_review.replace("[Review]: ", "").replace("NEEDS_IMPROVEMENT: ", "")
        
        # Get the original request
        original_request = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage) and "Previous attempt" not in msg.content:
                original_request = msg.content
                break
        
        # Get the last attempt
        last_attempt = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and "[Review]" not in msg.content:
                last_attempt = msg.content
                break
        
        # Create retry message with feedback
        retry_message = HumanMessage(content=f"""Previous attempt needs improvement. 

Original request: {original_request}

Your previous response: {last_attempt}

Feedback: {feedback}

Please provide an improved response that addresses the feedback.""")
        
        return {
            "messages": [retry_message]
        }
    
    def compile(self):
        """Compile and return the graph."""
        return self.graph.compile()


# Create and export the default app
def create_default_app():
    """Create a default review loop agent."""
    # Only create if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        # Return a dummy compiled graph for import purposes
        workflow = StateGraph(AgentState)
        workflow.add_node("dummy", lambda x: x)
        workflow.set_entry_point("dummy")
        workflow.add_edge("dummy", END)
        return workflow.compile()
    
    agent = ReviewLoopAgent(max_reviews=2)
    return agent.compile()


# Export the compiled graph as 'app'
app = create_default_app()


# Example usage and testing
if __name__ == "__main__":
    # Test the agent
    test_state = {
        "messages": [HumanMessage("What is 25 * 4 + 10?")],
        "review_count": 0,
        "max_reviews": 2
    }
    
    result = app.invoke(test_state)
    
    print("Final result:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and "[Review]" not in msg.content:
            print(msg.content)
            break
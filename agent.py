from typing import TypedDict, Literal, Callable, Any, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
import operator


class ReviewLoopState(TypedDict):
    """State for the review loop agent."""
    messages: list[BaseMessage]
    review_feedback: str
    iteration_count: int
    max_iterations: int
    is_approved: bool


class ReviewLoopAgent:
    """
    A generic review-loop agent that wraps any ReAct agent.
    
    The agent first runs a work agent to get initial output, then uses a review agent
    to evaluate that output. If the review is positive, it finishes. Otherwise, 
    it sends the feedback back to the work agent for another iteration.
    """
    
    def __init__(
        self, 
        work_agent: Callable[[Dict[str, Any]], Dict[str, Any]],
        review_llm: ChatOpenAI = None,
        max_iterations: int = 3
    ):
        """
        Initialize the review loop agent.
        
        Args:
            work_agent: The ReAct agent function that takes state and returns updated state
            review_llm: LLM for reviewing outputs (defaults to GPT-4o-mini)
            max_iterations: Maximum number of review iterations before forcing completion
        """
        self.work_agent = work_agent
        self.review_llm = review_llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.max_iterations = max_iterations
        
        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the review loop graph."""
        graph = StateGraph(ReviewLoopState)
        
        # Add nodes
        graph.add_node("work", self._work_node)
        graph.add_node("review", self._review_node)
        
        # Add edges
        graph.add_edge(START, "work")
        graph.add_edge("work", "review")
        graph.add_conditional_edges(
            "review",
            self._should_continue,
            {
                "continue": "work",
                "finish": END
            }
        )
        
        return graph
    
    def _work_node(self, state: ReviewLoopState) -> ReviewLoopState:
        """Execute the work agent."""
        # Add review feedback as context if this isn't the first iteration
        if state.get("review_feedback") and state.get("iteration_count", 0) > 0:
            feedback_message = HumanMessage(
                content=f"Previous review feedback: {state['review_feedback']}. Please address this feedback and improve your response."
            )
            work_messages = state["messages"] + [feedback_message]
        else:
            work_messages = state["messages"]
        
        # Call the work agent with current messages
        work_input = {"messages": work_messages}
        work_result = self.work_agent(work_input)
        
        # Update state with work agent's output
        return {
            **state,
            "messages": work_result["messages"],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "max_iterations": state.get("max_iterations", self.max_iterations)
        }
    
    def _review_node(self, state: ReviewLoopState) -> ReviewLoopState:
        """Review the work agent's output."""
        # Get the latest AI message (work agent's output)
        latest_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                latest_message = msg
                break
        
        if not latest_message:
            # No AI message found, approve to end the loop
            return {
                **state,
                "is_approved": True,
                "review_feedback": "No AI response found to review."
            }
        
        # Create review prompt
        review_prompt = f"""
        Please review the following response and determine if it adequately addresses the user's request.
        
        Original user request: {state['messages'][0].content if state['messages'] else 'Unknown request'}
        
        AI Response: {latest_message.content}
        
        Please evaluate:
        1. Does the response directly address the user's question or request?
        2. Is the response accurate and helpful?
        3. Is the response complete or are there missing elements?
        
        Respond with either:
        - "APPROVED: [brief reason why it's good]" if the response is satisfactory
        - "NEEDS_IMPROVEMENT: [specific feedback on what needs to be fixed/improved]" if it needs work
        """
        
        # Get review from LLM
        review_response = self.review_llm.invoke([HumanMessage(content=review_prompt)])
        review_content = review_response.content.strip()
        
        # Parse review
        is_approved = review_content.startswith("APPROVED:")
        feedback = review_content.split(":", 1)[1].strip() if ":" in review_content else review_content
        
        return {
            **state,
            "is_approved": is_approved,
            "review_feedback": feedback
        }
    
    def _should_continue(self, state: ReviewLoopState) -> Literal["continue", "finish"]:
        """Determine whether to continue the review loop or finish."""
        # Finish if approved or max iterations reached
        if state.get("is_approved", False) or state.get("iteration_count", 0) >= state.get("max_iterations", self.max_iterations):
            return "finish"
        return "continue"


def create_simple_react_agent():
    """
    Create a simple ReAct agent for demonstration purposes.
    This would normally be replaced with your actual ReAct agent.
    """
    def simple_agent(state_input):
        """A simple agent that just echoes the input with some processing."""
        messages = state_input.get("messages", [])
        
        if not messages:
            return {"messages": [AIMessage(content="I need a question or request to help you.")]}
        
        last_human_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_message = msg
                break
        
        if last_human_message:
            # Simple response - in a real scenario, this would be your ReAct agent
            response = f"I understand you're asking about: {last_human_message.content}. Let me think about this step by step and provide a helpful response."
            
            # Add the response to messages
            new_messages = messages + [AIMessage(content=response)]
            return {"messages": new_messages}
        
        return {"messages": messages + [AIMessage(content="I couldn't understand your request.")]}
    
    return simple_agent


# Create and export the app
def create_review_loop_app(work_agent=None, review_llm=None, max_iterations=3):
    """
    Create a review loop application.
    
    Args:
        work_agent: The ReAct agent to wrap (defaults to simple demo agent)
        review_llm: LLM for reviewing (defaults to GPT-4o-mini)
        max_iterations: Maximum review iterations
    """
    if work_agent is None:
        work_agent = create_simple_react_agent()
    
    review_agent = ReviewLoopAgent(
        work_agent=work_agent,
        review_llm=review_llm,
        max_iterations=max_iterations
    )
    
    return review_agent.app


# Export the compiled graph as 'app' (required by LangGraph)
app = create_review_loop_app()
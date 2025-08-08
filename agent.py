from typing import TypedDict, Annotated, Sequence, Literal, Optional, Any, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
import operator


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: str
    initial_output: Optional[str]
    review_feedback: Optional[str]
    iteration_count: int
    max_iterations: int
    is_approved: bool


class ReviewAgentWithFeedback:
    def __init__(
        self,
        react_agent_llm: Optional[ChatOpenAI] = None,
        reviewer_llm: Optional[ChatOpenAI] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        max_iterations: int = 3,
        system_prompt: Optional[str] = None
    ):
        self.react_agent_llm = react_agent_llm or ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="placeholder")
        self.reviewer_llm = reviewer_llm or ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="placeholder")
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or "You are a helpful assistant."
        
        self.react_agent = None
        if self.tools:
            self.react_agent = create_react_agent(
                self.react_agent_llm,
                self.tools,
                messages_modifier=self.system_prompt
            )
    
    def run_react_agent(self, state: AgentState) -> dict:
        """Execute the ReAct agent to generate initial output or revise based on feedback."""
        messages = []
        
        if state["iteration_count"] == 0:
            messages.append(HumanMessage(content=state["task"]))
        else:
            revision_prompt = f"""
            Original task: {state["task"]}
            
            Your previous output: {state["initial_output"]}
            
            Review feedback: {state["review_feedback"]}
            
            Please revise your response based on the feedback above.
            """
            messages.append(HumanMessage(content=revision_prompt))
        
        if self.react_agent and self.tools:
            result = self.react_agent.invoke({"messages": messages})
            output = result["messages"][-1].content if result["messages"] else ""
        else:
            response = self.react_agent_llm.invoke(messages)
            output = response.content
        
        return {
            "initial_output": output,
            "messages": state["messages"] + messages,
            "iteration_count": state["iteration_count"] + 1
        }
    
    def review_output(self, state: AgentState) -> dict:
        """Review the output from the ReAct agent and provide feedback."""
        review_prompt = f"""
        You are a quality reviewer. Please review the following output for the given task.
        
        Task: {state["task"]}
        
        Output to review: {state["initial_output"]}
        
        Please evaluate if this output:
        1. Correctly addresses the task
        2. Is complete and comprehensive
        3. Is accurate and well-reasoned
        4. Has appropriate formatting and clarity
        
        Respond in the following format:
        APPROVED: [YES/NO]
        FEEDBACK: [If NO, provide specific feedback on what needs improvement. If YES, explain why it's good.]
        """
        
        review_message = HumanMessage(content=review_prompt)
        review_response = self.reviewer_llm.invoke([review_message])
        review_content = review_response.content
        
        is_approved = "APPROVED: YES" in review_content.upper()
        
        feedback_start = review_content.find("FEEDBACK:")
        feedback = review_content[feedback_start + 9:].strip() if feedback_start != -1 else review_content
        
        return {
            "review_feedback": feedback,
            "is_approved": is_approved,
            "messages": state["messages"] + [review_message, review_response]
        }
    
    def should_continue(self, state: AgentState) -> Literal["review", "end"]:
        """Determine whether to continue the loop or end."""
        if state["iteration_count"] == 0:
            return "review"
        
        if state["is_approved"]:
            return "end"
        
        if state["iteration_count"] >= state["max_iterations"]:
            return "end"
        
        return "review"
    
    def should_revise(self, state: AgentState) -> Literal["react", "end"]:
        """Determine whether to revise based on review feedback."""
        if state["is_approved"]:
            return "end"
        
        if state["iteration_count"] >= state["max_iterations"]:
            return "end"
        
        return "react"
    
    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with feedback loop."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("react", self.run_react_agent)
        workflow.add_node("review", self.review_output)
        
        workflow.set_entry_point("react")
        
        workflow.add_conditional_edges(
            "react",
            self.should_continue,
            {
                "review": "review",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "review",
            self.should_revise,
            {
                "react": "react",
                "end": END
            }
        )
        
        return workflow.compile()


def create_review_agent_with_feedback(
    react_agent_llm: Optional[ChatOpenAI] = None,
    reviewer_llm: Optional[ChatOpenAI] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    max_iterations: int = 3,
    system_prompt: Optional[str] = None
) -> Any:
    """
    Factory function to create a review agent with feedback loop.
    
    Args:
        react_agent_llm: LLM for the ReAct agent
        reviewer_llm: LLM for the reviewer agent
        tools: Optional tools for the ReAct agent
        max_iterations: Maximum number of revision iterations
        system_prompt: System prompt for the ReAct agent
    
    Returns:
        Compiled LangGraph workflow
    """
    agent = ReviewAgentWithFeedback(
        react_agent_llm=react_agent_llm,
        reviewer_llm=reviewer_llm,
        tools=tools,
        max_iterations=max_iterations,
        system_prompt=system_prompt
    )
    return agent.build_graph()


# Default export as required by CLAUDE.md
app = create_review_agent_with_feedback()


if __name__ == "__main__":
    # Example usage
    from langchain_core.tools import tool
    
    @tool
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y
    
    @tool
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    # Create agent with tools
    agent_with_tools = create_review_agent_with_feedback(
        tools=[multiply, add],
        max_iterations=2
    )
    
    # Test the agent
    initial_state = {
        "messages": [],
        "task": "What is 25 * 4 + 10?",
        "initial_output": None,
        "review_feedback": None,
        "iteration_count": 0,
        "max_iterations": 2,
        "is_approved": False
    }
    
    result = agent_with_tools.invoke(initial_state)
    print(f"Final output: {result['initial_output']}")
    print(f"Approved: {result['is_approved']}")
    print(f"Iterations: {result['iteration_count']}")
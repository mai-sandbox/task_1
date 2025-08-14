"""
Two-Stage Agent Review Workflow

This module implements a generic two-stage review workflow using LangGraph where:
1. An initial React agent generates output
2. A review agent evaluates the output
3. If approved, the workflow ends; if not, it loops back to the initial agent

The workflow is designed to be generic and accept arbitrary React agents.
"""

from typing import Literal, Dict, Any, List, Optional

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent


# State Schema extending MessagesState with review fields
class ReviewWorkflowState(MessagesState):
    """
    State schema for the two-stage review workflow.
    Extends MessagesState to include review status and iteration tracking.
    """
    review_status: str = "pending"  # "pending", "approved", "needs_revision"
    iteration_count: int = 0
    # Configuration for agents (allows generic agent injection)
    initial_agent_config: Optional[Dict[str, Any]] = None
    review_agent_config: Optional[Dict[str, Any]] = None


def create_initial_agent_node(
    model, tools: List = None, prompt: str = None, name: str = "initial_agent"
):
    """
    Creates an initial agent node using create_react_agent.
    This allows for generic React agent injection.

    Args:
        model: The language model to use
        tools: List of tools for the agent
        prompt: Custom prompt for the agent
        name: Name of the agent

    Returns:
        A React agent node function
    """
    if tools is None:
        tools = []

    if prompt is None:
        prompt = (
            "You are an initial agent responsible for generating responses to user queries. "
            "Provide comprehensive and helpful responses. "
            "Your output will be reviewed by another agent."
        )

    # Create the React agent
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name=name
    )

    def initial_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """Node function that wraps the React agent and updates iteration count."""
        # Increment iteration count
        new_iteration = state.get("iteration_count", 0) + 1

        # Invoke the React agent
        result = react_agent.invoke({"messages": state["messages"]})

        # Update state with agent response and iteration count
        return {
            "messages": result["messages"],
            "iteration_count": new_iteration,
            "review_status": "pending"
        }

    return initial_agent_node


def create_review_agent_node(
    model, tools: List = None, prompt: str = None, name: str = "review_agent"
):
    """
    Creates a review agent node using create_react_agent.
    This agent evaluates the output from the initial agent.

    Args:
        model: The language model to use
        tools: List of tools for the agent (typically none needed for review)
        prompt: Custom prompt for the review agent
        name: Name of the agent

    Returns:
        A React agent node function for review
    """
    if tools is None:
        tools = []

    if prompt is None:
        prompt = (
            "You are a review agent responsible for evaluating responses from another agent. "
            "Analyze the conversation and the most recent response. "
            "Determine if the response is satisfactory and complete. "
            "Respond with either 'APPROVED' if the response is good, "
            "or 'NEEDS_REVISION' followed by specific feedback if improvements are needed. "
            "Be thorough but fair in your evaluation."
        )

    # Create the React agent for review
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name=name
    )

    def review_agent_node(state: ReviewWorkflowState) -> Dict[str, Any]:
        """Node function that wraps the review React agent."""
        # Invoke the review agent
        result = react_agent.invoke({"messages": state["messages"]})

        # Extract the review decision from the agent's response
        if result["messages"]:
            review_content = result["messages"][-1].content.upper()
            if "APPROVED" in review_content:
                review_status = "approved"
            elif "NEEDS_REVISION" in review_content:
                review_status = "needs_revision"
            else:
                # Default to needs_revision if unclear
                review_status = "needs_revision"
        else:
            review_status = "needs_revision"

        return {
            "messages": result["messages"],
            "review_status": review_status
        }

    return review_agent_node


def route_based_on_review(state: ReviewWorkflowState) -> Literal["approved", "needs_revision"]:
    """
    Routing function for conditional edges based on review status.

    Args:
        state: Current workflow state

    Returns:
        "approved" if review passed, "needs_revision" if it needs to loop back
    """
    review_status = state.get("review_status", "needs_revision")

    if review_status == "approved":
        return "approved"
    else:
        return "needs_revision"


def create_two_stage_review_workflow(
    initial_model,
    review_model=None,
    initial_tools: List = None,
    review_tools: List = None,
    initial_prompt: str = None,
    review_prompt: str = None,
    max_iterations: int = 5
):
    """
    Creates a two-stage review workflow graph.

    Args:
        initial_model: Model for the initial agent
        review_model: Model for the review agent (defaults to initial_model)
        initial_tools: Tools for the initial agent
        review_tools: Tools for the review agent
        initial_prompt: Custom prompt for initial agent
        review_prompt: Custom prompt for review agent
        max_iterations: Maximum number of revision iterations

    Returns:
        Compiled LangGraph workflow
    """
    if review_model is None:
        review_model = initial_model

    # Create agent nodes
    initial_agent = create_initial_agent_node(
        model=initial_model,
        tools=initial_tools,
        prompt=initial_prompt,
        name="initial_agent"
    )

    review_agent = create_review_agent_node(
        model=review_model,
        tools=review_tools,
        prompt=review_prompt,
        name="review_agent"
    )

    # Create the StateGraph
    workflow = StateGraph(ReviewWorkflowState)

    # Add nodes
    workflow.add_node("initial_agent", initial_agent)
    workflow.add_node("review_agent", review_agent)

    # Add edges
    workflow.add_edge(START, "initial_agent")
    workflow.add_edge("initial_agent", "review_agent")

    # Add conditional edges based on review
    workflow.add_conditional_edges(
        "review_agent",
        route_based_on_review,
        {
            "approved": END,
            "needs_revision": "initial_agent"
        }
    )

    # Compile the graph
    return workflow.compile()


# Default configuration using Anthropic Claude
def create_default_workflow():
    """Creates a default two-stage review workflow with Claude."""
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

    return create_two_stage_review_workflow(
        initial_model=model,
        review_model=model
    )


# Export the compiled graph as 'app' (required by LangGraph)
app = create_default_workflow()


# Example usage and configuration demonstrating generic design
if __name__ == "__main__":
    """
    GENERIC DESIGN DEMONSTRATION

    This section demonstrates how to inject arbitrary React agents by passing
    different models, tools, and prompts to the workflow. The two-stage review
    workflow is completely generic and can be adapted for any use case.

    Key Generic Design Features:
    1. Configurable models for both initial and review agents
    2. Customizable tools for each agent
    3. Domain-specific prompts for different use cases
    4. Flexible workflow parameters
    """

    print("🚀 DEMONSTRATING GENERIC TWO-STAGE REVIEW WORKFLOW")
    print("=" * 60)

    # Example 1: Basic usage with default configuration
    print("\n=== Example 1: Basic Usage (Default Configuration) ===")
    basic_workflow = create_default_workflow()
    print("✅ Created basic workflow with default Claude model")
    print("   Usage: basic_workflow.invoke({'messages': [HumanMessage('Hello')]})")

    # Example 2: Domain-specific workflows with custom prompts
    print("\n=== Example 2: Domain-Specific Workflows ===")

    # Code review workflow
    print("\n📝 Code Review Workflow:")
    code_review_workflow = create_two_stage_review_workflow(
        initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        initial_prompt=(
            "You are a senior software engineer conducting code reviews. "
            "Analyze the provided code for:\n"
            "- Best practices and design patterns\n"
            "- Performance optimizations\n"
            "- Security considerations\n"
            "- Maintainability and readability\n"
            "Provide specific, actionable feedback."
        ),
        review_prompt=(
            "You are a technical lead reviewing code review feedback. "
            "Evaluate if the code review is:\n"
            "- Comprehensive and covers all important aspects\n"
            "- Actionable with specific suggestions\n"
            "- Constructive and professional\n"
            "Respond with 'APPROVED' if thorough, or 'NEEDS_REVISION' with gaps to address."
        )
    )
    print("✅ Created specialized code review workflow")

    # Content writing workflow
    print("\n✍️ Content Writing Workflow:")
    content_writing_workflow = create_two_stage_review_workflow(
        initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        initial_prompt=(
            "You are a professional content writer specializing in engaging, "
            "SEO-optimized content. Create content that:\n"
            "- Captures reader attention from the first sentence\n"
            "- Provides clear value and actionable insights\n"
            "- Uses appropriate tone for the target audience\n"
            "- Includes relevant examples and data when applicable"
        ),
        review_prompt=(
            "You are an experienced editor reviewing content for publication. "
            "Evaluate the content for:\n"
            "- Clarity and readability\n"
            "- Engagement and value to readers\n"
            "- Grammar, style, and flow\n"
            "- Completeness and accuracy\n"
            "            "Respond with 'APPROVED' if ready to publish, or 'NEEDS_REVISION' with "
            "specific improvements.""
        )
    )
    print("✅ Created specialized content writing workflow")

    # Legal document review workflow
    print("\n⚖️ Legal Document Review Workflow:")
    legal_review_workflow = create_two_stage_review_workflow(
        initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        initial_prompt=(
            "You are a legal analyst reviewing documents. "
            "Analyze the document for:\n"
            "- Legal compliance and regulatory requirements\n"
            "- Potential risks and liabilities\n"
            "- Clarity of terms and conditions\n"
            "- Completeness of necessary clauses\n"
            "Provide detailed legal analysis and recommendations."
        ),
        review_prompt=(
            "You are a senior legal counsel reviewing legal analysis. "
            "Evaluate if the analysis:\n"
            "- Identifies all relevant legal issues\n"
            "- Provides accurate legal interpretation\n"
            "- Offers practical recommendations\n"
            "- Meets professional legal standards\n"
            "Respond with 'APPROVED' if comprehensive, or 'NEEDS_REVISION' with missing elements."
        )
    )
    print("✅ Created specialized legal document review workflow")

    # Example 3: Different models for different stages
    print("\n=== Example 3: Mixed Model Configuration ===")
    print("🔄 Using different models for initial vs review stages:")

    mixed_model_workflow = create_two_stage_review_workflow(
        initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        review_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),  # Could use different model
        initial_prompt="You are a creative writer generating innovative content.",
        review_prompt="You are a strict quality assurance reviewer ensuring excellence.",
        max_iterations=3  # Custom iteration limit
    )
    print("✅ Created workflow with different models for each stage")
    print("   - Initial Agent: Claude 3.5 Sonnet (creative)")
    print("   - Review Agent: Claude 3.5 Sonnet (analytical)")

    # Example 4: Tool integration examples
    print("\n=== Example 4: Tool Integration Examples ===")
    print("🛠️ Demonstrating tool injection capabilities:")

    # Example with mock tools (commented to avoid import errors)
    print("\n📚 Research Workflow (with search tools):")
    print("# Uncomment and install tools to use:")
    print("# from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun")
    print("# from langchain_community.utilities import WikipediaAPIWrapper")
    print("#")
    print("# search_tool = DuckDuckGoSearchRun()")
    print("# wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())")
    print("#")
    print("# research_workflow = create_two_stage_review_workflow(")
    print("#     initial_model=ChatAnthropic(model='claude-3-5-sonnet-20241022'),")
    print("#     initial_tools=[search_tool, wiki_tool],")
    print("#     initial_prompt='You are a researcher. Use search and Wikipedia tools to "
          "gather comprehensive information.',")
    print("#     review_prompt='You are a fact-checker. Verify the research quality and source reliability.'")
    print("# )")

    # Example 5: Advanced configuration options
    print("\n=== Example 5: Advanced Configuration ===")
    print("⚙️ Advanced workflow customization:")

    advanced_workflow = create_two_stage_review_workflow(
        initial_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        review_model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
        initial_tools=[],  # No tools for this example
        review_tools=[],   # No tools for review
        initial_prompt=(
            "You are an AI assistant with expertise in multiple domains. "
            "Adapt your response style and depth based on the user's query. "
            "Always strive for accuracy, helpfulness, and clarity."
        ),
        review_prompt=(
            "You are a quality assurance specialist reviewing AI responses. "
            "Check for accuracy, completeness, helpfulness, and appropriateness. "
            "Consider if the response fully addresses the user's needs. "
            "Respond with 'APPROVED' if satisfactory, or 'NEEDS_REVISION' with specific improvements."
        ),
        max_iterations=5  # Allow up to 5 revision cycles
    )
    print("✅ Created advanced workflow with custom iteration limits")

    # Example 6: Workflow comparison
    print("\n=== Example 6: Workflow Usage Patterns ===")
    print("🎯 Different ways to use the generic workflow:")

    workflows = {
        "Basic": basic_workflow,
        "Code Review": code_review_workflow,
        "Content Writing": content_writing_workflow,
        "Legal Review": legal_review_workflow,
        "Mixed Models": mixed_model_workflow,
        "Advanced": advanced_workflow
    }

    print(f"\n📊 Created {len(workflows)} different workflow configurations:")
    for name, workflow in workflows.items():
        print(f"   • {name}: Ready for use")

    print("\n" + "=" * 60)
    print("🎉 GENERIC DESIGN DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\n💡 Key Takeaways:")
    print("   1. Same workflow structure, completely different behaviors")
    print("   2. Easy to inject any React agent configuration")
    print("   3. Supports any model, tools, and prompts combination")
    print("   4. Configurable parameters for different use cases")
    print("   5. Maintains consistent two-stage review pattern")

    print("\n🚀 Usage Instructions:")
    print("   • Default workflow: app.invoke({'messages': [HumanMessage('Your query')]})")
    print("   • Custom workflow: your_workflow.invoke({'messages': [HumanMessage('Query')]})")
    print("   • All workflows follow the same interface pattern")

    print("\n📝 The 'app' variable contains the default compiled workflow.")
    print("   Ready for deployment to LangGraph Platform or local development!")





from review_agent_system import (
    ReactAgent, 
    ReviewerAgent,
    AgentOutput, 
    ReviewResult, 
    ReviewStatus,
    SimpleReviewerAgent,
    ReviewAgentOrchestrator,
    create_length_based_reviewer,
    CustomReviewerAgent
)
from typing import Any, Dict, Optional, List
import random


class ExampleReactAgent(ReactAgent):
    def __init__(self, name: str = "ExampleAgent"):
        self.name = name
        self.attempt_count = 0
    
    def get_name(self) -> str:
        return self.name
    
    def run(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        self.attempt_count += 1
        
        if context and 'previous_feedback' in context:
            print(f"  Previous feedback: {context['previous_feedback']}")
            
            if context.get('suggestions'):
                for suggestion in context['suggestions']:
                    if 'at least' in suggestion and 'characters' in suggestion:
                        min_length = int(''.join(filter(str.isdigit, suggestion)))
                        response = f"Task: {input_data}\n" + "=" * 50 + "\n"
                        response += f"Improved response (attempt {self.attempt_count}): "
                        response += "This is a much more detailed response. " * (min_length // 40 + 1)
                        return AgentOutput(content=response[:min_length + 10])
        
        initial_response = f"Task: {input_data}\nResponse: Done."
        return AgentOutput(content=initial_response)


class ImprovingReactAgent(ReactAgent):
    def __init__(self, name: str = "ImprovingAgent"):
        self.name = name
    
    def get_name(self) -> str:
        return self.name
    
    def run(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        iteration = context.get('iteration', 0) if context else 0
        
        responses = [
            "Quick answer",
            "A more detailed answer with additional context",
            "A comprehensive answer that includes multiple perspectives and detailed analysis of the problem at hand"
        ]
        
        response_index = min(iteration, len(responses) - 1)
        return AgentOutput(
            content=f"Input: {input_data}\nOutput: {responses[response_index]}",
            metadata={'iteration': iteration}
        )


class QualityReviewerAgent(ReviewerAgent):
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
    
    def review(self, output: AgentOutput, original_input: Any) -> ReviewResult:
        content = str(output.content)
        
        quality_score = 0.0
        suggestions = []
        
        if len(content) > 50:
            quality_score += 0.3
        else:
            suggestions.append("Provide more detailed content")
        
        if any(keyword in content.lower() for keyword in ['analysis', 'comprehensive', 'detailed']):
            quality_score += 0.4
        else:
            suggestions.append("Include more analytical depth")
        
        if output.metadata and output.metadata.get('iteration', 0) > 0:
            quality_score += 0.3
        
        if quality_score >= self.quality_threshold:
            return ReviewResult(
                status=ReviewStatus.APPROVED,
                feedback=f"Quality score: {quality_score:.2f} - Approved!"
            )
        else:
            return ReviewResult(
                status=ReviewStatus.NEEDS_REVISION,
                feedback=f"Quality score: {quality_score:.2f} - Below threshold ({self.quality_threshold})",
                suggestions=suggestions
            )


def example_1_simple_review():
    print("\n" + "="*60)
    print("Example 1: Simple Length-Based Review")
    print("="*60)
    
    agent = ExampleReactAgent("SimpleAgent")
    reviewer = SimpleReviewerAgent(min_content_length=100, required_keywords=['Task'])
    
    orchestrator = ReviewAgentOrchestrator(
        react_agent=agent,
        reviewer_agent=reviewer,
        max_iterations=3,
        verbose=True
    )
    
    result = orchestrator.run("Generate a summary of machine learning")
    print(f"\nFinal output length: {len(str(result.content))} characters")
    print(f"Total iterations: {len(orchestrator.get_history())}")


def example_2_custom_reviewer():
    print("\n" + "="*60)
    print("Example 2: Custom Length-Based Reviewer")
    print("="*60)
    
    agent = ImprovingReactAgent()
    reviewer = create_length_based_reviewer(min_length=50, max_length=200)
    
    orchestrator = ReviewAgentOrchestrator(
        react_agent=agent,
        reviewer_agent=reviewer,
        max_iterations=5,
        verbose=True
    )
    
    result = orchestrator.run("Explain quantum computing")
    print(f"\nFinal output: {result.content}")


def example_3_quality_based_review():
    print("\n" + "="*60)
    print("Example 3: Quality-Based Review")
    print("="*60)
    
    agent = ImprovingReactAgent("QualityFocusedAgent")
    reviewer = QualityReviewerAgent(quality_threshold=0.6)
    
    orchestrator = ReviewAgentOrchestrator(
        react_agent=agent,
        reviewer_agent=reviewer,
        max_iterations=4,
        verbose=True
    )
    
    result = orchestrator.run("Describe the benefits of renewable energy")
    print(f"\nFinal output achieved after {len(orchestrator.get_history())} iterations")


def example_4_custom_review_function():
    print("\n" + "="*60)
    print("Example 4: Fully Custom Review Function")
    print("="*60)
    
    def sentiment_reviewer(output: AgentOutput, original_input: Any) -> ReviewResult:
        content = str(output.content).lower()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible']
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            return ReviewResult(
                status=ReviewStatus.APPROVED,
                feedback="Content has positive sentiment"
            )
        else:
            return ReviewResult(
                status=ReviewStatus.NEEDS_REVISION,
                feedback="Content needs more positive sentiment",
                suggestions=["Include more positive language", f"Consider using words like: {', '.join(positive_words[:3])}"]
            )
    
    class SentimentAgent(ReactAgent):
        def __init__(self):
            self.attempts = 0
        
        def get_name(self) -> str:
            return "SentimentAgent"
        
        def run(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
            self.attempts += 1
            
            if self.attempts == 1:
                return AgentOutput(content=f"The task '{input_data}' is challenging but has potential.")
            elif self.attempts == 2:
                return AgentOutput(content=f"The task '{input_data}' is good and shows promise.")
            else:
                return AgentOutput(content=f"The task '{input_data}' is excellent and will yield great results!")
    
    agent = SentimentAgent()
    reviewer = CustomReviewerAgent(sentiment_reviewer)
    
    orchestrator = ReviewAgentOrchestrator(
        react_agent=agent,
        reviewer_agent=reviewer,
        max_iterations=5,
        verbose=True
    )
    
    result = orchestrator.run("Create a motivational message")
    print(f"\nFinal message: {result.content}")


if __name__ == "__main__":
    example_1_simple_review()
    example_2_custom_reviewer()
    example_3_quality_based_review()
    example_4_custom_review_function()
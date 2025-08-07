from typing import Any, Dict, List, Optional, Protocol, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    ERROR = "error"


@dataclass
class AgentOutput:
    content: Any
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReviewResult:
    status: ReviewStatus
    feedback: str
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ReactAgent(ABC):
    @abstractmethod
    def run(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class ReviewerAgent(ABC):
    @abstractmethod
    def review(self, output: AgentOutput, original_input: Any) -> ReviewResult:
        pass


class SimpleReviewerAgent(ReviewerAgent):
    def __init__(self, 
                 min_content_length: int = 10,
                 required_keywords: List[str] = None,
                 max_error_tolerance: int = 0):
        self.min_content_length = min_content_length
        self.required_keywords = required_keywords or []
        self.max_error_tolerance = max_error_tolerance
    
    def review(self, output: AgentOutput, original_input: Any) -> ReviewResult:
        if output.error:
            return ReviewResult(
                status=ReviewStatus.ERROR,
                feedback=f"Output contains error: {output.error}",
                suggestions=["Fix the error and try again"]
            )
        
        content_str = str(output.content)
        suggestions = []
        
        if len(content_str) < self.min_content_length:
            suggestions.append(f"Content should be at least {self.min_content_length} characters")
        
        missing_keywords = []
        for keyword in self.required_keywords:
            if keyword.lower() not in content_str.lower():
                missing_keywords.append(keyword)
        
        if missing_keywords:
            suggestions.append(f"Missing required keywords: {', '.join(missing_keywords)}")
        
        if suggestions:
            return ReviewResult(
                status=ReviewStatus.NEEDS_REVISION,
                feedback="Output needs improvement",
                suggestions=suggestions
            )
        
        return ReviewResult(
            status=ReviewStatus.APPROVED,
            feedback="Output meets all requirements"
        )


class ReviewAgentOrchestrator:
    def __init__(self, 
                 react_agent: ReactAgent,
                 reviewer_agent: ReviewerAgent,
                 max_iterations: int = 3,
                 verbose: bool = True):
        self.react_agent = react_agent
        self.reviewer_agent = reviewer_agent
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.iteration_history: List[Tuple[AgentOutput, ReviewResult]] = []
    
    def _log(self, message: str):
        if self.verbose:
            logger.info(message)
    
    def run(self, input_data: Any, initial_context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        self.iteration_history = []
        context = initial_context or {}
        
        for iteration in range(self.max_iterations):
            self._log(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            if iteration > 0:
                last_review = self.iteration_history[-1][1]
                context['previous_feedback'] = last_review.feedback
                context['suggestions'] = last_review.suggestions
                context['iteration'] = iteration
            
            self._log(f"Running {self.react_agent.get_name()} agent...")
            output = self.react_agent.run(input_data, context)
            
            self._log(f"Reviewing output...")
            review_result = self.reviewer_agent.review(output, input_data)
            
            self.iteration_history.append((output, review_result))
            
            self._log(f"Review status: {review_result.status.value}")
            self._log(f"Feedback: {review_result.feedback}")
            
            if review_result.status == ReviewStatus.APPROVED:
                self._log("✅ Output approved!")
                return output
            
            if review_result.status == ReviewStatus.ERROR:
                self._log(f"❌ Error detected: {review_result.feedback}")
                if iteration == self.max_iterations - 1:
                    return output
            
            if review_result.suggestions:
                self._log(f"Suggestions for improvement:")
                for suggestion in review_result.suggestions:
                    self._log(f"  - {suggestion}")
        
        self._log(f"\n⚠️ Maximum iterations ({self.max_iterations}) reached without approval")
        return self.iteration_history[-1][0] if self.iteration_history else AgentOutput(
            content=None, 
            error="No output generated"
        )
    
    def get_history(self) -> List[Tuple[AgentOutput, ReviewResult]]:
        return self.iteration_history


class CustomReviewerAgent(ReviewerAgent):
    def __init__(self, review_function):
        self.review_function = review_function
    
    def review(self, output: AgentOutput, original_input: Any) -> ReviewResult:
        return self.review_function(output, original_input)


def create_length_based_reviewer(min_length: int, max_length: int) -> ReviewerAgent:
    def review_func(output: AgentOutput, original_input: Any) -> ReviewResult:
        content_length = len(str(output.content))
        
        if content_length < min_length:
            return ReviewResult(
                status=ReviewStatus.NEEDS_REVISION,
                feedback=f"Content too short ({content_length} chars)",
                suggestions=[f"Expand content to at least {min_length} characters"]
            )
        
        if content_length > max_length:
            return ReviewResult(
                status=ReviewStatus.NEEDS_REVISION,
                feedback=f"Content too long ({content_length} chars)",
                suggestions=[f"Reduce content to at most {max_length} characters"]
            )
        
        return ReviewResult(
            status=ReviewStatus.APPROVED,
            feedback=f"Content length ({content_length}) is within acceptable range"
        )
    
    return CustomReviewerAgent(review_func)
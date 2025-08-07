from typing import Any, Dict, Optional, Callable, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ReviewStatus(Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    

@dataclass
class AgentOutput:
    content: Any
    metadata: Optional[Dict[str, Any]] = None
    iteration: int = 0
    

@dataclass
class ReviewResult:
    status: ReviewStatus
    feedback: str
    suggestions: Optional[List[str]] = None
    

class BaseAgent(ABC):
    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> AgentOutput:
        pass
        

class ReactAgent(BaseAgent):
    def __init__(self, 
                 thought_process: Optional[Callable] = None,
                 action_executor: Optional[Callable] = None,
                 observation_handler: Optional[Callable] = None):
        self.thought_process = thought_process or self._default_thought
        self.action_executor = action_executor or self._default_action
        self.observation_handler = observation_handler or self._default_observation
        
    def _default_thought(self, input_data: Any) -> str:
        return f"Processing input: {input_data}"
        
    def _default_action(self, thought: str) -> Any:
        return {"action": "process", "thought": thought}
        
    def _default_observation(self, action_result: Any) -> Any:
        return {"observation": action_result}
        
    def run(self, input_data: Any, previous_feedback: Optional[str] = None, **kwargs) -> AgentOutput:
        context = {"input": input_data}
        if previous_feedback:
            context["previous_feedback"] = previous_feedback
            
        thought = self.thought_process(context)
        action_result = self.action_executor(thought)
        observation = self.observation_handler(action_result)
        
        output = {
            "thought": thought,
            "action": action_result,
            "observation": observation,
            "final_output": observation
        }
        
        return AgentOutput(
            content=output,
            metadata={"react_steps": ["thought", "action", "observation"]}
        )


class ReviewerAgent(BaseAgent):
    def __init__(self, 
                 review_criteria: Optional[List[Callable]] = None,
                 approval_threshold: float = 0.8):
        self.review_criteria = review_criteria or [self._default_criterion]
        self.approval_threshold = approval_threshold
        
    def _default_criterion(self, output: AgentOutput) -> Tuple[float, str]:
        if output.content:
            return 1.0, "Output contains content"
        return 0.0, "Output is empty"
        
    def run(self, input_data: AgentOutput, **kwargs) -> ReviewResult:
        scores = []
        feedback_items = []
        suggestions = []
        
        for criterion in self.review_criteria:
            score, feedback = criterion(input_data)
            scores.append(score)
            feedback_items.append(feedback)
            
            if score < self.approval_threshold:
                suggestions.append(f"Improve: {feedback}")
                
        avg_score = sum(scores) / len(scores) if scores else 0
        
        status = ReviewStatus.APPROVED if avg_score >= self.approval_threshold else ReviewStatus.NEEDS_REVISION
        
        return ReviewResult(
            status=status,
            feedback=" | ".join(feedback_items),
            suggestions=suggestions if suggestions else None
        )


class DummyAgent:
    def __init__(self, 
                 react_agent: ReactAgent,
                 reviewer_agent: Optional[ReviewerAgent] = None,
                 max_iterations: int = 3,
                 verbose: bool = False):
        self.react_agent = react_agent
        self.reviewer_agent = reviewer_agent or ReviewerAgent()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []
        
    def run(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        iteration = 0
        current_input = input_data
        previous_feedback = None
        
        while iteration < self.max_iterations:
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
                
            react_output = self.react_agent.run(
                current_input, 
                previous_feedback=previous_feedback,
                **kwargs
            )
            react_output.iteration = iteration
            
            if self.verbose:
                print(f"React Agent Output: {react_output.content}")
                
            review_result = self.reviewer_agent.run(react_output)
            
            if self.verbose:
                print(f"Review Status: {review_result.status.value}")
                print(f"Feedback: {review_result.feedback}")
                
            self.history.append({
                "iteration": iteration,
                "react_output": react_output,
                "review_result": review_result
            })
            
            if review_result.status == ReviewStatus.APPROVED:
                if self.verbose:
                    print(f"\n✓ Output approved after {iteration + 1} iteration(s)")
                    
                return {
                    "final_output": react_output.content,
                    "iterations": iteration + 1,
                    "status": "approved",
                    "history": self.history
                }
                
            previous_feedback = review_result.feedback
            if review_result.suggestions:
                previous_feedback += " Suggestions: " + ", ".join(review_result.suggestions)
                
            iteration += 1
            
        if self.verbose:
            print(f"\n✗ Max iterations ({self.max_iterations}) reached without approval")
            
        return {
            "final_output": self.history[-1]["react_output"].content if self.history else None,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached",
            "history": self.history
        }
        
    def reset_history(self):
        self.history = []
from dummy_agent import DummyAgent, ReactAgent, ReviewerAgent, AgentOutput, ReviewStatus
from typing import Any, Dict, Tuple
import json


class MathReactAgent(ReactAgent):
    def __init__(self):
        super().__init__()
        
    def _thought_process(self, context: Dict[str, Any]) -> str:
        problem = context["input"]
        if "previous_feedback" in context:
            return f"Reconsidering problem '{problem}' based on feedback: {context['previous_feedback']}"
        return f"Analyzing mathematical problem: {problem}"
        
    def _action_executor(self, thought: str) -> Any:
        if "2+2" in thought:
            return {"calculation": "2+2", "result": 4}
        elif "10*5" in thought:
            return {"calculation": "10*5", "result": 50}
        elif "complex" in thought.lower():
            return {"calculation": "complex", "result": "needs_more_work"}
        else:
            return {"calculation": "unknown", "result": "error"}
            
    def _observation_handler(self, action_result: Any) -> Any:
        if action_result.get("result") == "error":
            return {"status": "failed", "answer": None}
        elif action_result.get("result") == "needs_more_work":
            return {"status": "incomplete", "answer": "working on it..."}
        else:
            return {"status": "complete", "answer": action_result["result"]}
            
    def run(self, input_data: Any, previous_feedback: str = None, **kwargs) -> AgentOutput:
        context = {"input": input_data}
        if previous_feedback:
            context["previous_feedback"] = previous_feedback
            
        thought = self._thought_process(context)
        action_result = self._action_executor(thought)
        observation = self._observation_handler(action_result)
        
        if previous_feedback and "incomplete" in previous_feedback:
            observation["status"] = "complete"
            observation["answer"] = 42
            
        output = {
            "thought": thought,
            "action": action_result,
            "observation": observation,
            "final_answer": observation.get("answer")
        }
        
        return AgentOutput(
            content=output,
            metadata={"problem_type": "mathematical"}
        )


class MathReviewer(ReviewerAgent):
    def __init__(self):
        super().__init__(
            review_criteria=[
                self._check_completeness,
                self._check_correctness,
                self._check_format
            ],
            approval_threshold=0.7
        )
        
    def _check_completeness(self, output: AgentOutput) -> Tuple[float, str]:
        content = output.content
        if content.get("observation", {}).get("status") == "complete":
            return 1.0, "Solution is complete"
        elif content.get("observation", {}).get("status") == "incomplete":
            return 0.3, "Solution is incomplete"
        return 0.0, "No valid solution found"
        
    def _check_correctness(self, output: AgentOutput) -> Tuple[float, str]:
        answer = output.content.get("final_answer")
        if answer is not None and answer != "error":
            return 1.0, f"Answer provided: {answer}"
        return 0.0, "No valid answer"
        
    def _check_format(self, output: AgentOutput) -> Tuple[float, str]:
        required_keys = ["thought", "action", "observation"]
        if all(key in output.content for key in required_keys):
            return 1.0, "Output format is correct"
        return 0.5, "Output format is partially correct"


def example_simple():
    print("=" * 50)
    print("Example 1: Simple Math Problem (Should Pass)")
    print("=" * 50)
    
    math_agent = MathReactAgent()
    reviewer = MathReviewer()
    dummy_agent = DummyAgent(
        react_agent=math_agent,
        reviewer_agent=reviewer,
        max_iterations=3,
        verbose=True
    )
    
    result = dummy_agent.run("What is 2+2?")
    print(f"\nFinal Result: {json.dumps(result['final_output'], indent=2)}")
    print(f"Status: {result['status']}")
    print(f"Total Iterations: {result['iterations']}")


def example_with_revision():
    print("\n" + "=" * 50)
    print("Example 2: Complex Problem (Needs Revision)")
    print("=" * 50)
    
    math_agent = MathReactAgent()
    reviewer = MathReviewer()
    dummy_agent = DummyAgent(
        react_agent=math_agent,
        reviewer_agent=reviewer,
        max_iterations=3,
        verbose=True
    )
    
    result = dummy_agent.run("Solve this complex equation")
    print(f"\nFinal Result: {json.dumps(result['final_output'], indent=2)}")
    print(f"Status: {result['status']}")
    print(f"Total Iterations: {result['iterations']}")


class CustomReactAgent(ReactAgent):
    def __init__(self, process_func):
        super().__init__()
        self.process_func = process_func
        
    def run(self, input_data: Any, previous_feedback: str = None, **kwargs) -> AgentOutput:
        result = self.process_func(input_data, previous_feedback)
        return AgentOutput(content=result)


def example_custom_agent():
    print("\n" + "=" * 50)
    print("Example 3: Custom Agent with Lambda")
    print("=" * 50)
    
    def text_processor(text, feedback):
        if feedback and "too short" in feedback:
            return {"text": text.upper() + "!!!", "length": len(text) + 3}
        return {"text": text.upper(), "length": len(text)}
    
    def text_reviewer_criteria(output: AgentOutput) -> Tuple[float, str]:
        content = output.content
        if content.get("length", 0) > 10:
            return 1.0, "Text is long enough"
        return 0.3, "Text is too short"
    
    custom_agent = CustomReactAgent(text_processor)
    custom_reviewer = ReviewerAgent(
        review_criteria=[text_reviewer_criteria],
        approval_threshold=0.8
    )
    
    dummy_agent = DummyAgent(
        react_agent=custom_agent,
        reviewer_agent=custom_reviewer,
        max_iterations=2,
        verbose=True
    )
    
    result = dummy_agent.run("hello")
    print(f"\nFinal Result: {json.dumps(result['final_output'], indent=2)}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    example_simple()
    example_with_revision()
    example_custom_agent()
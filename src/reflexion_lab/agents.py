from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import os
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .memory_compression import compress_reflection_memory

# Support both mock and real runtime
USE_MOCK = os.getenv("USE_MOCK_RUNTIME", "false").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

if USE_MOCK:
    from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer as mock_actor, evaluator as mock_evaluator, reflector as mock_reflector
    
    def actor_answer(example, attempt_id, agent_type, reflection_memory):
        answer = mock_actor(example, attempt_id, agent_type, reflection_memory)
        token_estimate = 320 + (attempt_id * 65) + (120 if agent_type == "reflexion" else 0)
        latency_ms = 160 + (attempt_id * 40) + (90 if agent_type == "reflexion" else 0)
        return answer, token_estimate, latency_ms
    
    def evaluator(example, answer):
        judge = mock_evaluator(example, answer)
        return judge, 150, 80
    
    def reflector(example, attempt_id, judge, answer):
        reflection = mock_reflector(example, attempt_id, judge)
        return reflection, 200, 100
elif LLM_PROVIDER == "minimax":
    from .minimax_runtime import actor_answer, evaluator, reflector
    FAILURE_MODE_BY_QID = {}
else:  # default to openai
    from .llm_runtime import actor_answer, evaluator, reflector
    FAILURE_MODE_BY_QID = {}

def classify_failure_mode(example: QAExample, judge, answer: str, reflections: list[ReflectionEntry]) -> str:
    """Classify the type of failure based on the error pattern"""
    if judge.score == 1:
        return "none"
    
    # Check for looping (same reflection pattern)
    if len(reflections) >= 2:
        if reflections[-1].lesson == reflections[-2].lesson:
            return "looping"
    
    # Check for reflection overfit (too many attempts with no improvement)
    if len(reflections) >= 3:
        return "reflection_overfit"
    
    # Check failure reason patterns
    reason_lower = judge.reason.lower()
    if "incomplete" in reason_lower or "first hop" in reason_lower or "never completed" in reason_lower:
        return "incomplete_multi_hop"
    if "wrong" in reason_lower and ("entity" in reason_lower or "second" in reason_lower):
        return "entity_drift"
    
    # Classify based on question difficulty and content
    if example.difficulty == "hard":
        return "complex_reasoning_failure"
    elif example.difficulty == "medium":
        return "multi_hop_failure"
    
    # Classify based on answer content patterns
    if len(answer.split()) > 10:  # Long verbose answer
        return "verbose_incorrect_answer"
    elif len(answer.split()) <= 2:  # Very short answer
        return "insufficient_reasoning"
    
    return "wrong_final_answer"

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    adaptive_max_attempts: bool = False  # Bonus feature
    use_memory_compression: bool = True  # Bonus feature
    
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        
        # Adaptive max attempts based on difficulty (bonus feature)
        max_attempts = self.max_attempts
        if self.adaptive_max_attempts and self.agent_type == "reflexion":
            if example.difficulty == "easy":
                max_attempts = 2
            elif example.difficulty == "medium":
                max_attempts = 3
            else:  # hard
                max_attempts = 4
        
        for attempt_id in range(1, max_attempts + 1):
            # Apply memory compression if enabled (bonus feature)
            active_memory = reflection_memory
            if self.use_memory_compression and len(reflection_memory) > 2:
                active_memory = compress_reflection_memory(reflection_memory, max_items=2)
            
            # Get answer from Actor
            answer, actor_tokens, actor_latency = actor_answer(
                example, attempt_id, self.agent_type, active_memory
            )
            
            # Evaluate the answer
            judge, eval_tokens, eval_latency = evaluator(example, answer)
            
            # Calculate total tokens and latency for this attempt
            attempt_tokens = actor_tokens + eval_tokens
            attempt_latency = actor_latency + eval_latency
            
            # Create trace for this attempt
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=attempt_tokens,
                latency_ms=attempt_latency
            )
            
            final_answer = answer
            final_score = judge.score
            
            # If correct, we're done
            if judge.score == 1:
                traces.append(trace)
                break
            
            # Reflexion logic: Generate reflection if this is a reflexion agent and not the last attempt
            if self.agent_type == "reflexion" and attempt_id < max_attempts:
                reflection, refl_tokens, refl_latency = reflector(
                    example, attempt_id, judge, answer
                )
                
                # Update trace with reflection
                trace.reflection = reflection
                trace.token_estimate += refl_tokens
                trace.latency_ms += refl_latency
                
                # Add to reflections list
                reflections.append(reflection)
                
                # Update reflection memory for next attempt
                reflection_text = f"Attempt {attempt_id} failed.\nReason: {reflection.failure_reason}\nLesson: {reflection.lesson}\nStrategy: {reflection.next_strategy}"
                reflection_memory.append(reflection_text)
            
            traces.append(trace)
        
        # Calculate totals
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        
        # Classify failure mode
        if USE_MOCK:
            failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        else:
            failure_mode = classify_failure_mode(example, judge, final_answer, reflections)
        
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces
        )

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1, adaptive_max_attempts=False, use_memory_compression=False)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, adaptive_max_attempts: bool = False, use_memory_compression: bool = True) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, adaptive_max_attempts=adaptive_max_attempts, use_memory_compression=use_memory_compression)

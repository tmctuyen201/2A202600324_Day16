"""Real LLM runtime using OpenAI API"""
from __future__ import annotations
import json
import os
import time
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .utils import normalize_answer

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model configuration
ACTOR_MODEL = "gpt-4o-mini"
EVALUATOR_MODEL = os.getenv("JUDGE_MODEL_1", "gpt-4o")
REFLECTOR_MODEL = "gpt-4o-mini"

def count_tokens_estimate(text: str) -> int:
    """Rough token estimation: ~4 chars per token"""
    return len(text) // 4

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str]
) -> tuple[str, int, int]:
    """
    Generate an answer using the Actor agent.
    Returns: (answer, token_count, latency_ms)
    """
    # Build context string
    context_str = "\n\n".join([
        f"Title: {chunk.title}\n{chunk.text}"
        for chunk in example.context
    ])
    
    # Build user prompt
    user_prompt = f"""Question: {example.question}

Context:
{context_str}"""
    
    # Add reflection memory if available
    if reflection_memory:
        reflection_str = "\n\n".join([
            f"Reflection {i+1}:\n{refl}"
            for i, refl in enumerate(reflection_memory)
        ])
        user_prompt += f"\n\nPrevious Reflections:\n{reflection_str}\n\nUse the reflections above to improve your answer."
    
    # Call OpenAI API
    start_time = time.time()
    response = client.chat.completions.create(
        model=ACTOR_MODEL,
        messages=[
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )
    latency_ms = int((time.time() - start_time) * 1000)
    
    answer = response.choices[0].message.content.strip()
    token_count = response.usage.total_tokens
    
    return answer, token_count, latency_ms

def evaluator(
    example: QAExample,
    answer: str
) -> tuple[JudgeResult, int, int]:
    """
    Evaluate an answer using the Evaluator agent.
    Returns: (judge_result, token_count, latency_ms)
    """
    user_prompt = f"""Question: {example.question}
Gold Answer: {example.gold_answer}
Predicted Answer: {answer}

Evaluate if the predicted answer is correct."""
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"}
    )
    latency_ms = int((time.time() - start_time) * 1000)
    
    token_count = response.usage.total_tokens
    
    try:
        result_dict = json.loads(response.choices[0].message.content)
        judge_result = JudgeResult(**result_dict)
    except Exception as e:
        # Fallback to simple normalization-based evaluation
        is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
        judge_result = JudgeResult(
            score=1 if is_correct else 0,
            reason=f"Normalization-based evaluation. Error parsing LLM response: {e}",
            missing_evidence=[],
            spurious_claims=[] if is_correct else [answer]
        )
    
    return judge_result, token_count, latency_ms

def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
    answer: str
) -> tuple[ReflectionEntry, int, int]:
    """
    Generate reflection using the Reflector agent.
    Returns: (reflection_entry, token_count, latency_ms)
    """
    # Build context string
    context_str = "\n\n".join([
        f"Title: {chunk.title}\n{chunk.text}"
        for chunk in example.context
    ])
    
    user_prompt = f"""Question: {example.question}
Gold Answer: {example.gold_answer}
Your Answer (Attempt {attempt_id}): {answer}

Context:
{context_str}

Evaluation Result:
- Score: {judge.score}
- Reason: {judge.reason}
- Missing Evidence: {', '.join(judge.missing_evidence) if judge.missing_evidence else 'None'}
- Spurious Claims: {', '.join(judge.spurious_claims) if judge.spurious_claims else 'None'}

Reflect on why this attempt failed and provide guidance for the next attempt."""
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=REFLECTOR_MODEL,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        response_format={"type": "json_object"}
    )
    latency_ms = int((time.time() - start_time) * 1000)
    
    token_count = response.usage.total_tokens
    
    try:
        result_dict = json.loads(response.choices[0].message.content)
        result_dict["attempt_id"] = attempt_id
        reflection = ReflectionEntry(**result_dict)
    except Exception as e:
        # Fallback reflection
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="The answer was incorrect. Need to be more careful with the reasoning.",
            next_strategy="Re-read the context carefully and ensure all hops are completed."
        )
    
    return reflection, token_count, latency_ms

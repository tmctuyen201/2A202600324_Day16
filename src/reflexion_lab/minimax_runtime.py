"""Minimax LLM runtime integration"""
from __future__ import annotations
import json
import os
import time
import requests
from typing import Any
from dotenv import load_dotenv
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .utils import normalize_answer

# Load environment variables
load_dotenv()

# Minimax API configuration
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"

# Model configuration - using abab6.5s-chat (Minimax 2.7)
ACTOR_MODEL = "abab6.5s-chat"
EVALUATOR_MODEL = "abab6.5s-chat"
REFLECTOR_MODEL = "abab6.5s-chat"

def count_tokens_estimate(text: str) -> int:
    """Rough token estimation: ~2 chars per token for Chinese/English mix"""
    return len(text) // 2

def call_minimax_api(messages: list[dict], temperature: float = 0.7, max_tokens: int = 2000) -> tuple[str, int, int]:
    """
    Call Minimax API and return response, token count, and latency.
    
    Args:
        messages: List of message dicts with role and content
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        (response_text, token_count, latency_ms)
    """
    if not MINIMAX_API_KEY or not MINIMAX_GROUP_ID:
        raise ValueError("MINIMAX_API_KEY and MINIMAX_GROUP_ID must be set in .env file")
    
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": ACTOR_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "tokens_to_generate": max_tokens
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            MINIMAX_BASE_URL,
            headers=headers,
            json=payload,
            params={"GroupId": MINIMAX_GROUP_ID},
            timeout=60
        )
        response.raise_for_status()
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        result = response.json()
        
        # Extract response text
        if "choices" in result and len(result["choices"]) > 0:
            response_text = result["choices"][0]["message"]["content"]
        else:
            response_text = result.get("reply", "")
        
        # Estimate tokens (Minimax doesn't always return usage)
        if "usage" in result:
            token_count = result["usage"].get("total_tokens", count_tokens_estimate(response_text))
        else:
            # Estimate based on input + output
            input_text = " ".join([m["content"] for m in messages])
            token_count = count_tokens_estimate(input_text + response_text)
        
        return response_text, token_count, latency_ms
        
    except requests.exceptions.RequestException as e:
        print(f"Minimax API error: {e}")
        # Return fallback
        latency_ms = int((time.time() - start_time) * 1000)
        return f"Error: {str(e)}", 100, latency_ms

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
    
    # Call Minimax API
    messages = [
        {"role": "system", "content": ACTOR_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    
    answer, token_count, latency_ms = call_minimax_api(messages, temperature=0.7, max_tokens=150)
    
    return answer.strip(), token_count, latency_ms

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

Evaluate if the predicted answer is correct. Return JSON format."""
    
    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    
    response_text, token_count, latency_ms = call_minimax_api(messages, temperature=0, max_tokens=300)
    
    try:
        # Try to parse JSON from response
        # Sometimes the response might have extra text, so we extract JSON
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result_dict = json.loads(json_str)
            judge_result = JudgeResult(**result_dict)
        else:
            raise ValueError("No JSON found in response")
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

Reflect on why this attempt failed and provide guidance for the next attempt. Return JSON format."""
    
    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    
    response_text, token_count, latency_ms = call_minimax_api(messages, temperature=0.7, max_tokens=300)
    
    try:
        # Try to parse JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result_dict = json.loads(json_str)
            result_dict["attempt_id"] = attempt_id
            reflection = ReflectionEntry(**result_dict)
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        # Fallback reflection
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="The answer was incorrect. Need to be more careful with the reasoning.",
            next_strategy="Re-read the context carefully and ensure all hops are completed."
        )
    
    return reflection, token_count, latency_ms

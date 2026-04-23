"""Memory compression for reflection history - Bonus Feature"""
from __future__ import annotations
from typing import List

def compress_reflection_memory(reflections: List[str], max_items: int = 2) -> List[str]:
    """
    Compress reflection memory to prevent context overflow.
    Keeps only the most recent reflections and summarizes older ones.
    
    Args:
        reflections: List of reflection strings
        max_items: Maximum number of full reflections to keep
    
    Returns:
        Compressed list of reflections
    """
    if len(reflections) <= max_items:
        return reflections
    
    # Keep the most recent reflections
    recent = reflections[-max_items:]
    
    # Summarize older reflections
    older_count = len(reflections) - max_items
    summary = f"[Summary of {older_count} earlier reflections: Multiple attempts were made with various strategies. Key lessons learned about multi-hop reasoning and entity verification.]"
    
    return [summary] + recent

def extract_key_lessons(reflections: List[str]) -> str:
    """
    Extract key lessons from multiple reflections.
    
    Args:
        reflections: List of reflection strings
    
    Returns:
        Summarized key lessons
    """
    if not reflections:
        return ""
    
    # Simple extraction - in a real implementation, this could use LLM
    lessons = []
    for refl in reflections:
        if "Lesson:" in refl:
            lesson_part = refl.split("Lesson:")[1].split("Strategy:")[0].strip()
            lessons.append(lesson_part)
    
    if not lessons:
        return "Previous attempts have failed. Need to be more careful."
    
    # Return unique lessons
    unique_lessons = list(dict.fromkeys(lessons))
    return " | ".join(unique_lessons[:3])  # Keep top 3 unique lessons

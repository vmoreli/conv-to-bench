from __future__ import annotations
from typing import Optional, TypedDict, List
from pydantic import BaseModel

class Feedback(TypedDict, total=False):
    msgs_positive_feedback: List[int]
    msgs_negative_feedback: List[int]

class ChecklistItem(TypedDict, total=False):
    reasoning: str
    requirement: str
    check: Optional[bool]

class Checklist(TypedDict, total=False):
    items: List[ChecklistItem]
    
class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add(self, other: dict | TokenUsage):
        if isinstance(other, TokenUsage):
            other = other.model_dump()
        self.prompt_tokens += other.get("prompt_tokens", 0)
        self.completion_tokens += other.get("completion_tokens", 0)
        self.total_tokens += other.get("total_tokens", 0)
        self.total_cost_usd += other.get("total_cost_usd", 0.0)
        return self


# ---------------------------------------------------------
# Estado global
# ---------------------------------------------------------

class BenchmarkBuilderState(TypedDict, total=False):
    # Extraction
    raw_conversation: str
    conversation: str
    is_programming_related: bool
    instruction: Optional[str]
    is_valid_instruction: Optional[bool]
    code: Optional[str]

    # Feedback Evaluation
    feedback: Optional[Feedback]
    checklist: Optional[Checklist]
    fulfilled: Optional[bool]
    retries_evaluation: Optional[int]

    # TOKEN USAGE
    token_usage: Optional[TokenUsage]
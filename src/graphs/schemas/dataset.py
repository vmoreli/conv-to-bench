from pydantic import BaseModel, Field
from typing import List

class ProgrammingRelatedSchema(BaseModel):
    programming_related: bool = Field(
        ...,
        description=(
            "True if the conversation is about programming or code "
            "(e.g., languages, algorithms, libraries, debugging, etc.). "
            "False if the conversation is about any other topic."
        )
    )

class InstructionSchema(BaseModel):
    instruction: str = Field(
        ...,
        description="A clear, concise, and well-structured restatement of the user's original coding request. If the request involves modifying an existing code, include that code the instruction."
    )

class IsValidInstruction(BaseModel):
    is_valid_instruction: bool = Field(
        ...,
        description="Indicates wether the instruction is valid or not according to the guidelines."
    )

class FeedbackSchema(BaseModel):
    """
    Schema to capture implicit user feedback from a conversation.
    Feedback is always extracted from the *user's* messages in reaction
    to the *assistant's* responses.
    """
    msgs_positive_feedback: List[int] = Field(
        None,
        description=(
            "A list containing the numeric IDs of the **user's** messages that express positive feedback. "
            "Positive feedback indicates that the assistant's previous response was helpful, correct, or solved the problem. "
        )
    )
    msgs_negative_feedback: List[int] = Field(
        None,
        description=(
            "A list containing the numeric IDs of the **user's** messages that express negative feedback. "
            "Negative feedback suggests the assistant's previous response was unhelpful, incorrect, incomplete, or confusing. "
        )
    )

class ChecklistReqItem(BaseModel):
    requirement: str = Field(
        ...,
        description=(
            "Item of the evaluation checklist."
        )
    )

class ChecklistReqSchema(BaseModel):
    items: List[ChecklistReqItem]
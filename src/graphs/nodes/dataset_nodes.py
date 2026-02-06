import json
from langgraph.types import Command
from typing import Literal
from langgraph.graph import END
from pydantic import BaseModel
from src.llm.client import call_llm


from src.graphs.prompts.generator import (
    # system prompts
    is_programming_related_system_prompt,
    extract_instruction_system_prompt,
    filter_instruction_system_prompt,
    extract_feedback_system_prompt,
    generate_checklist_system_prompt,
    
    # user prompts
    is_programming_related_user_prompt,
    extract_instruction_user_prompt,
    filter_instruction_user_prompt,
    extract_feedback_user_prompt,
    generate_checklist_user_prompt,
)

from src.graphs.schemas.dataset import (
    ProgrammingRelatedSchema,
    InstructionSchema, 
    IsValidInstruction,
    FeedbackSchema,
    ChecklistReqSchema
)

from src.graphs.states import (
    Feedback,
    ChecklistItem,
    Checklist,
)

from src.data.extraction import first_user_message


# ---------------------------------------------------------
# Node to determine if a conversation is code-related
# ---------------------------------------------------------
def is_programming_related_node(state) -> Command[Literal["extract_instruction", END]]:
    conversation_raw = state["raw_conversation"]
    first_msg = first_user_message(conversation_raw)

    user_prompt = is_programming_related_user_prompt.format(
        first_message=first_msg
    )

    response_raw = call_llm(
        system_prompt=is_programming_related_system_prompt,
        user_prompt=user_prompt,
        output_schema=ProgrammingRelatedSchema
    )
    response_json = json.loads(response_raw['content'].text)
    token_usage = response_raw['usage']

    is_programming_related = response_json["programming_related"]

    goto = "extract_instruction" if is_programming_related else END

    return Command(
        goto=goto,
        update={
            "is_programming_related": is_programming_related,
            "token_usage": state["token_usage"].add(token_usage),
        },
    )


# ---------------------------------------------------------
# Node to extract instruction from a conversation
# ---------------------------------------------------------
def extract_instruction_node(state) -> Command[Literal["extract_code"]]:
    conversation = state["conversation"]

    user_prompt = extract_instruction_user_prompt.format(conversation=conversation)
    
    response_raw = call_llm(
        user_prompt=user_prompt,
        system_prompt=extract_instruction_system_prompt,
        output_schema=InstructionSchema
    )
    response_json = json.loads(response_raw['content'])
    token_usage = response_raw['usage']

    instruction = response_json["instruction"]

    # goto = "extract_code"
    goto = "feedback_eval"

    return Command(
        goto=goto,
        update={
            "instruction": instruction,
            "token_usage": state["token_usage"].add(token_usage),
        },
    )

# ---------------------------------------------------------
# Node to determine if a conversation has feedback,
# and in which messages the feedback appears
# ---------------------------------------------------------
def identify_feedback_node(state) -> Command[Literal["generate_checklist"]]:
    conversation = state["conversation"]

    user_prompt = extract_feedback_user_prompt.format(conversation=conversation)
    response_raw = call_llm(
        system_prompt=extract_feedback_system_prompt,
        user_prompt=user_prompt,
        output_schema=FeedbackSchema
    )
    response_json = json.loads(response_raw['content'])
    token_usage = response_raw['usage']

    msgs_positive_feedback = response_json["msgs_positive_feedback"]
    msgs_negative_feedback = response_json["msgs_negative_feedback"]

    feedback: Feedback = {
        "msgs_negative_feedback": msgs_negative_feedback,
        "msgs_positive_feedback": msgs_positive_feedback
    }

    goto = "generate_checklist"
    # goto = END


    return Command(
        goto=goto,
        update={"feedback": feedback, "token_usage": state["token_usage"].add(token_usage)},
    )

# ---------------------------------------------------------
# Node to create a requirements checklist based on
# instruction and negative feedback
# ---------------------------------------------------------
def generate_checklist_node(state) -> Command[Literal[END]]:
    conversation = state["conversation"]
    instruction = state["instruction"]
    feedback = state["feedback"]

    # Extract ids from feedback
    pos_ids = feedback["msgs_positive_feedback"] or []
    neg_ids = feedback["msgs_negative_feedback"] or []

    # Convert lists to strings
    positive_ids_str = ', '.join(map(str, pos_ids)) if pos_ids else "None"
    negative_ids_str = ', '.join(map(str, neg_ids)) if neg_ids else "None"

    # Fill prompt
    user_prompt = generate_checklist_user_prompt.format(
        conversation=conversation,
        instruction=instruction,
        positive_feedback_ids=positive_ids_str,
        negative_feedback_ids=negative_ids_str
    )

    response_raw = call_llm(
        system_prompt=generate_checklist_system_prompt,
        user_prompt=user_prompt,
        output_schema=ChecklistReqSchema
    )
    response_json = json.loads(response_raw['content'])
    token_usage = response_raw['usage']


    checklist: Checklist = {
        "items": []
    }
    for item in response_json["items"]:
        item: ChecklistItem = {
            "requirement": item["requirement"]
        }
        checklist["items"].append(item)

    goto = END

    return Command(
        goto=goto,
        update={"checklist": checklist, "token_usage": state["token_usage"].add(token_usage)},
    )

def filter_instructions_node(state):
    instruction = state["instruction"]
    
    user_prompt = filter_instruction_user_prompt.format(
        instruction=instruction
    )

    response_raw = call_llm(
        system_prompt=filter_instruction_system_prompt,
        user_prompt=user_prompt,
        output_schema=IsValidInstruction
    )
    content = response_raw['content']
    if isinstance(content, BaseModel):
        response_json = content.model_dump()  # convert ChecksSchema -> dict
    else:
        response_json = json.loads(content)
    token_usage = response_raw['usage']

    is_valid_instruction = response_json["is_valid_instruction"]

    # Keep commented, as it's just debug
    # if is_valid_instruction:
    #     goto = "feedback_eval"
    # else:
    #     goto = END

    goto = END

    return Command(
        goto=goto,
        update={"is_valid_instruction": is_valid_instruction, "token_usage": state["token_usage"].add(token_usage)},
    )
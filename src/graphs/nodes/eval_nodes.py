import json
from langgraph.types import Command
from langgraph.graph import END
from typing import Literal
from pydantic import BaseModel
from src.llm.client import call_llm

from src.graphs.prompts.evaluator import (
    # system prompts
    eval_solution_system_prompt,
    
    # user prompts
    eval_solution_user_prompt,
)

from src.graphs.prompts.evaluator import (
    # system prompts 
    eval_solution_system_prompt,
    # user prompt
    eval_solution_user_prompt,
)

from src.graphs.schemas.evaluation import (
    ChecksSchema,
)

from src.graphs.states import (
    ChecklistItem,
    Checklist,
    TokenUsage,
)


def chunked(iterable, n):
    """Divide a list or iterable into chunks of size n."""
    items = list(iterable)
    for i in range(0, len(items), n):
        yield items[i:i + n]

# ---------------------------------------------------------
# Node to evaluate the solution based on the checklist
# ---------------------------------------------------------
def eval_solution_node(state) -> Command[Literal[END]]:
    checklist = state["checklist"]
    code = state["code"]
    retries = 0
    max_retries = 3

    goto = END

    reqs_I = []
    reqs_F = []

    # Initialize local token accumulator
    total_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }

    # Populates lists of instruction and feedback requirements
    for item in checklist["items"]:
        req = item["requirement"]
        if req.startswith("[I"):
            reqs_I.append(item)
        elif req.startswith("[F"):
            reqs_F.append(item)
        else:
            raise RuntimeError(f"Unknown requirement: {req}")

    checklist_result_items = []

    # Helper function to evaluate a batch (chunk)
    def evaluate_chunk(chunk_items):
        nonlocal retries
        while retries < max_retries:
            # Create a "clean" version of requirements without [I...] or [F...] prefixes
            cleaned_items = []
            for item in chunk_items:
                cleaned_req = item["requirement"]
                if cleaned_req.startswith("["):
                    cleaned_req = cleaned_req.split("]", 1)[-1].strip()
                cleaned_item = {**item, "requirement": cleaned_req}
                cleaned_items.append(cleaned_item)

            partial_checklist = {"items": cleaned_items}

            user_prompt = eval_solution_user_prompt.format(
                checklist=json.dumps(partial_checklist),
                code=code
            )

            response_raw = call_llm(
                system_prompt=eval_solution_system_prompt,
                user_prompt=user_prompt,
                output_schema=ChecksSchema
            )

            # Conta tokens dessa chamada e acumula
            token_usage = response_raw['usage']
            for k in total_token_usage:
                total_token_usage[k] += token_usage.get(k, 0)

            content = response_raw['content']
            if isinstance(content, BaseModel):
                response_json = content.model_dump()  # convert ChecksSchema -> dict
            else:
                response_json = json.loads(content)

            items = response_json["items"]

            if len(items) != len(chunk_items):
                retries += 1
                continue  # try the same block again

            # assemble part of the resulting checklist
            partial_result = [
                ChecklistItem(
                    reasoning=item["reasoning"],
                    requirement=req["requirement"],
                    check=item["check"]
                )
                for req, item in zip(chunk_items, items)
            ]
            return partial_result

        return []  # if exceeds max_retries, return empty

    # Process in blocks of 3
    for chunk in chunked(reqs_I, 3):
        checklist_result_items.extend(evaluate_chunk(chunk))

    for chunk in chunked(reqs_F, 3):
        checklist_result_items.extend(evaluate_chunk(chunk))

    fulfilled = all(item["check"] for item in checklist_result_items) if checklist_result_items else False

    checklist_result = Checklist(items=checklist_result_items)

    print(total_token_usage)

    return Command(
        goto=goto,
        update={
            "checklist": checklist_result,
            "fulfilled": fulfilled,
            "retries_evaluation": retries,
            "token_usage": state.get("token_usage", TokenUsage()).add(total_token_usage)
        },
    )
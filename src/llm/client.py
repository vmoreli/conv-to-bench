import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from config import CONV_HIST_DIR, PRICES

# Load environment variables
load_dotenv()

# Configurable Model
# SELECTED_MODEL = "gemini-2.0-flash-001"
# SELECTED_MODEL = "gpt-4.1-nano-2025-04-14"
SELECTED_MODEL = "gemini-2.5-flash"
# SELECTED_MODEL = "gpt-4o-mini"
# SELECTED_MODEL = "gpt-5-mini"

# Initialize clients on demand
if SELECTED_MODEL.startswith("gemini"):
    from google import genai
    from google.genai import types
    client = genai.Client()
elif SELECTED_MODEL.startswith(("gpt", "o1")):
    from openai import OpenAI
    client = OpenAI()
else:
    raise ValueError(f"Model '{SELECTED_MODEL}' not supported (use 'gemini-*' or 'gpt-*').")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
) -> Dict[str, Any]:
    """
    Automatically calls the Gemini or OpenAI model based on the configuration.
    Returns a dictionary with { content, usage }.
    """
    if SELECTED_MODEL.startswith("gemini"):
        response = _call_gemini(system_prompt, user_prompt, output_schema)
    else:
        response = _call_openai(system_prompt, user_prompt, output_schema)
    return response


# -------------------------------------------------------------------
# GEMINI IMPLEMENTATION
# -------------------------------------------------------------------
def _call_gemini(system_prompt: str, user_prompt: str, output_schema: Optional[Type[BaseModel]] = None):
    from google.genai import types

    response = client.models.generate_content(
        model=SELECTED_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json" if output_schema else "text/plain",
            response_schema=output_schema,
        ),
    )

    _save_call_log(system_prompt, user_prompt, response)
    usage_stats = _count_tokens_gemini(response)
    
    # Handle response content
    content = response.text if hasattr(response, "text") else str(response)

    if output_schema:
        try:
            content = output_schema.model_validate_json(content)
        except ValidationError as e:
            print("[WARN] Failed to validate schema (Gemini):", e)
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass # Return raw string if JSON fails

    return {"content": content, "usage": usage_stats}


def _count_tokens_gemini(response) -> Dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    
    usage_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }

    if usage:
        usage_stats["prompt_tokens"] = usage.prompt_token_count
        usage_stats["completion_tokens"] = usage.candidates_token_count
        usage_stats["total_tokens"] = usage.total_token_count

        input_price = PRICES[SELECTED_MODEL].get("input", 0.0)
        output_price = PRICES[SELECTED_MODEL].get("output", 0.0)

        input_cost = usage.prompt_token_count * input_price / 1_000_000
        output_cost = usage.candidates_token_count * output_price / 1_000_000
        usage_stats["total_cost_usd"] = input_cost + output_cost

    return usage_stats


# -------------------------------------------------------------------
# OPENAI IMPLEMENTATION
# -------------------------------------------------------------------
def _call_openai(system_prompt: str, user_prompt: str, output_schema: Optional[Type[BaseModel]] = None):
    """
    Generic call for OpenAI models.
    """
    try:
        response = client.chat.completions.parse(
            model=SELECTED_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=output_schema,
        )

        content = response.choices[0].message.content
        
        if output_schema:
            try:
                content = output_schema.model_validate_json(content)
            except ValidationError as e:
                print("[WARN] Failed to validate schema (OpenAI):", e)
                try:
                    content = json.loads(content)
                except Exception:
                    pass  # Keep raw content if not valid JSON

        _save_call_log(system_prompt, user_prompt, response)
        usage_stats = _count_tokens_openai(response)

        return {"content": content, "usage": usage_stats}

    except Exception as e:
        print(f"[ERROR] Failed to call model {SELECTED_MODEL}: {e}")
        raise


def _count_tokens_openai(response) -> Dict[str, Any]:
    usage = response.usage

    usage_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }

    if usage:
        usage_stats["prompt_tokens"] = usage.prompt_tokens
        usage_stats["completion_tokens"] = usage.completion_tokens
        usage_stats["total_tokens"] = usage.total_tokens

        # Check for reasoning tokens (o1/o3 models)
        if hasattr(usage, "completion_tokens_details"):
             usage_stats["reasoning_tokens"] = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)

        input_price = PRICES[SELECTED_MODEL].get("input", 0.0)
        output_price = PRICES[SELECTED_MODEL].get("output", 0.0)

        input_cost = usage.prompt_tokens * input_price / 1_000_000
        output_cost = usage.completion_tokens * output_price / 1_000_000
        usage_stats["total_cost_usd"] = input_cost + output_cost

    return usage_stats


# -------------------------------------------------------------------
# COMMON UTILITIES
# -------------------------------------------------------------------
def _save_call_log(system_prompt: str, user_prompt: str, response: Any):
    """Saves the call log (with timestamp and unique ID) to disk."""
    try:
        os.makedirs(CONV_HIST_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"llm_call_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        file_path = os.path.join(CONV_HIST_DIR, file_name)

        log_data = {
            "timestamp": timestamp,
            "model": SELECTED_MODEL,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        # Safely convert the response object to dict/str
        if hasattr(response, "model_dump"):
            log_data["raw_response"] = response.model_dump()
        elif hasattr(response, "to_dict"):
            log_data["raw_response"] = response.to_dict()
        else:
            log_data["raw_response"] = str(response)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save LLM call history: {e}")
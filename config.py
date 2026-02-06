MODEL="gemini-2.5-flash"

LMSYS_CODING_CONVS_PATH='topic_modeling/topic_model_lmsys/coding_convs.json'
WILDCHAT_CODING_CONVS_PATH='topic_modeling/topic_model_wildchat/coding_convs.json'

LMSYS_DS_PATH='data/lmsys/train'
WILDCHAT_DS_PATH='data/wildchat/train'

LMSYS_FILTERED_PATH='data/lmsys_coding/'
WILDCHAT_FILTERED_PATH='data/wildchat_coding/'

TRACES_PATH='traces/'
CONV_HIST_DIR='log_llm_calls/'
BENCH_PATH='bench/'

PRICES={
    "gemini-2.0-flash-001": {
        "input": 0.1,
        "output": 0.4
    },
    "gemini-2.5-flash": {
        "input": 0.3,
        "output": 2.5
    },
    "gpt-4.1-nano-2025-04-14": {
        "input": 0.1,
        "output": 0.4
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.6
    },
    "gpt-5-mini": {
        "input": 0.25,
        "output": 2 
    },
}

# config.py
from pathlib import Path

# Base Definitions
EVALUATED_MODELS = [
        "deepseek-coder-6.7b-instruct", 
        "CodeQwen1.5-7B-Chat", 
        "CodeLlama-13b-Instruct-hf", 
        "codegemma-7b-it", 
        "CodeLlama-7b-Instruct-hf", 
        "Qwen2.5-Coder-7B-Instruct",
        "gemini-2.0-flash-001",
        "gpt-4.1-nano-2025-04-14"
    ]

EVALUATOR_MODELS = [
    "gemini-2.5-flash",
    "gpt-5-mini",
    "deepseek"
]
BASE_DIR = Path("bench/evals_unique_run")

# Dynamic generation via Dictionary Comprehension
RESULT_PATHS = {
    m_eval: {
        m_judge: BASE_DIR / m_eval / "runs" / f"run_{m_judge}" / f"{m_eval}_results.json"
        for m_judge in EVALUATOR_MODELS
    }
    for m_eval in EVALUATED_MODELS
}
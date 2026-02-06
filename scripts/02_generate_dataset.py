import os
import json
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

# Project Imports
from src.data.loader import load_datasets
from src.graphs.workflows import dataset_app 
from src.utils.tracking import aggregate_token_usage
from src.utils.text import format_conversation

def process_single_conversation(item: Dict[str, Any], source_name: str):
    """
    Processes a single conversation item through the LangGraph pipeline.
    """
    # Determine ID based on source
    key_map = {
        "lmsys": "conversation_id",
        "wildchat": "conversation_hash",
    }
    
    conv_id = item.get(key_map.get(source_name, "id"))
    conversation_key = f"{conv_id}_{source_name}"

    # Extract raw messages
    raw_conversation = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in item.get("conversation", [])
        if "role" in msg and "content" in msg
    ]
    
    # Format for LLM input
    formatted_text = format_conversation(raw_conversation)

    # Invoke LangGraph Pipeline
    # Note: pipeline.invoke inputs must match your State definition
    result = dataset_app.invoke(
        {"conversation_text": formatted_text, "raw_messages": raw_conversation}
    )

    # Safely extract usage (LangGraph usually returns it in metadata or state)
    # Adjust 'token_usage' key based on your State implementation
    token_usage = result.get("token_usage", {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0
    })

    return conversation_key, result, len(raw_conversation), token_usage

def process_dataset_batch(
    dataset, 
    source_name: str, 
    results_store: dict, 
    metadata_store: dict, 
    global_usage: dict, 
    max_workers: int = 4
):
    total = len(dataset)
    print(f"🚀 Starting processing for '{source_name}' with {total} items...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_conversation, line, source_name): i 
            for i, line in enumerate(dataset)
        }

        for i, future in enumerate(as_completed(futures)):
            try:
                conv_key, response, num_turns, usage = future.result()
                
                results_store[conv_key] = response
                metadata_store[conv_key] = num_turns
                aggregate_token_usage(global_usage, usage)
                
            except Exception as e:
                idx = futures[future]
                print(f"⚠️ Error processing item {idx}: {e}")
            
            # Simple progress log
            if (i + 1) % 10 == 0:
                print(f"[{source_name}] Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

def save_results(results, metadata, token_usage, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    combined_data = {
        k: {"extracted_data": v, "num_turns": metadata.get(k, 0)}
        for k, v in results.items()
    }

    # Save Responses
    resp_path = os.path.join(run_dir, "extracted_dataset.json")
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2, default=str)

    # Save Metrics
    token_path = os.path.join(run_dir, "token_usage.json")
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(token_usage, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved to: {run_dir}")
    print(f"📊 Total Conversations: {len(results)}")
    print(f"💰 Total Cost: ${token_usage.get('total_cost_usd', 0):.4f}")

def main(args):
    # 1. Load Data
    lmsys_ds, wildchat_ds = load_datasets()
    
    results_store = {}
    metadata_store = {}
    
    global_usage = {
        "prompt_tokens": 0, "completion_tokens": 0, 
        "total_tokens": 0, "total_cost_usd": 0.0
    }

    # 2. Process Datasets
    # You can add logic here to choose which dataset to process via args
    process_dataset_batch(lmsys_ds, "lmsys", results_store, metadata_store, global_usage, args.workers)
    process_dataset_batch(wildchat_ds, "wildchat", results_store, metadata_store, global_usage, args.workers)

    # 3. Save
    save_results(results_store, metadata_store, global_usage, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset generation pipeline.")
    parser.add_argument("--output-dir", type=str, default="output/traces", help="Directory to save results")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    main(args)
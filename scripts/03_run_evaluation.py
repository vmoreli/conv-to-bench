import json
import time
import argparse
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

# Project Imports
from src.graphs.workflows import evaluator_app
from src.utils.serialization import sanitize_for_json
from src.utils.tracking import aggregate_token_usage

# Configuration Constants
DEFAULT_CHECKPOINT_INTERVAL = 20
DEFAULT_TIMEOUT_SECONDS = 600

def save_progress_checkpoint(results, token_usage, file_path: Path):
    """Saves output safely, attempting UTF-8 then falling back to ASCII."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "results": sanitize_for_json(results),
        "token_usage": sanitize_for_json(token_usage),
    }
    
    try:
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"💾 Checkpoint saved: {file_path.name}")
    except Exception as e:
        print(f"⚠️ UTF-8 Save failed: {e}. Falling back to ASCII.")
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="ascii", errors="replace")

def evaluate_single_item(conversation_id, response_text, instruction, checklist_items):
    """
    Invokes the evaluation graph for a single item.
    """
    # Invoke LangGraph Evaluation Pipeline
    result = evaluator_app.invoke({
        "instruction": instruction,
        "model_response": response_text, # Adjust key matches your state
        "checklist": checklist_items
    })
    
    # Extract usage from state/result
    token_usage = result.get("token_usage", {})
    return conversation_id, result, token_usage

def run_evaluation_loop(
    responses_path: Path,
    checklist_path: Path,
    output_path: Path,
    model_name: str,
    workers: int,
    resume: bool
):
    # 1. Load Input Data
    print(f"📂 Loading responses from: {responses_path}")
    all_responses = json.loads(responses_path.read_text(encoding="utf-8"))
    
    print(f"📂 Loading checklists from: {checklist_path}")
    checklists = json.loads(checklist_path.read_text(encoding="utf-8"))

    # 2. Checkpoint / Resume Logic
    results = {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0}
    
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        # Find latest checkpoint like filename_100.json
        checkpoint_files = sorted(output_dir.glob(f"{output_path.stem}_*.json"))
        if checkpoint_files:
            # Sort by number in filename
            last_ckpt = max(checkpoint_files, key=lambda p: int(re.findall(r"_(\d+)\.json", p.name)[0]))
            print(f"🔁 Resuming from checkpoint: {last_ckpt.name}")
            
            ckpt_data = json.loads(last_ckpt.read_text(encoding="utf-8"))
            results = ckpt_data.get("results", {})
            total_usage = ckpt_data.get("token_usage", total_usage)

    processed_ids = set(results.keys())

    # 3. Prepare Work Items
    work_items = {}
    for conv_id, check_data in checklists.items():
        if conv_id in processed_ids:
            continue
            
        # Extract the specific model response
        model_response_data = all_responses.get(conv_id, {}).get("responses", {}).get(model_name)
        
        if not model_response_data:
            # print(f"⚠️ Response missing for {conv_id} / {model_name}")
            continue

        work_items[conv_id] = {
            "response": model_response_data,
            "instruction": check_data.get("instruction"),
            "checklist": check_data.get("checklist")
        }

    total_items = len(work_items)
    print(f"🚀 Starting evaluation for {model_name}: {total_items} items remaining.")

    if total_items == 0:
        print("✅ Nothing to process.")
        return

    # 4. Parallel Execution Loop
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        
        # Submit all tasks
        for conv_id, data in work_items.items():
            if not all([data["response"], data["instruction"], data["checklist"]]):
                continue
                
            fut = executor.submit(
                evaluate_single_item, 
                conv_id, 
                data["response"], 
                data["instruction"], 
                data["checklist"]
            )
            futures[fut] = conv_id

        # Process as completed
        completed_count = len(processed_ids)
        
        for future in as_completed(futures):
            conv_id = futures[future]
            completed_count += 1
            
            try:
                # Wait for result with timeout
                _, result_state, usage = future.result(timeout=DEFAULT_TIMEOUT_SECONDS)
                
                # Store result (adjust structure as needed)
                results[conv_id] = {
                    "evaluation_result": result_state.get("evaluation"), # e.g. Pass/Fail
                    "reasoning": result_state.get("reasoning"),
                    "score": result_state.get("score")
                }
                
                aggregate_token_usage(total_usage, dict(usage))
                
            except TimeoutError:
                print(f"⏰ Timeout processing {conv_id}")
                results[conv_id] = {"error": "timeout"}
            except Exception as e:
                print(f"❌ Error in {conv_id}: {e}")
                results[conv_id] = {"error": str(e)}

            # Progress & Checkpointing
            if completed_count % DEFAULT_CHECKPOINT_INTERVAL == 0:
                print(f"📈 Progress: {completed_count} items processed...")
                ckpt_path = output_dir / f"{output_path.stem}_{completed_count}.json"
                save_progress_checkpoint(results, total_usage, ckpt_path)

    # 5. Final Save
    save_progress_checkpoint(results, total_usage, output_path)
    elapsed = time.time() - start_time
    print(f"✅ Finished {model_name} in {elapsed:.1f}s. Saved to {output_path}")

def main(args):
    # Determine models to evaluate
    if args.models:
        models = args.models
    else:
        # Default fallback list
        models = ["codellama/CodeLlama-7b-Instruct-hf"]

    for model in models:
        model_short_name = model.split("/")[-1]
        
        base_dir = Path(args.output_dir) / model_short_name
        base_dir.mkdir(parents=True, exist_ok=True)
        
        for run_idx in range(1, args.runs + 1):
            run_file = base_dir / f"run_{run_idx}_eval_results.json"
            
            if run_file.exists() and not args.force:
                print(f"⏭️ Run {run_idx} for {model_short_name} exists. Skipping.")
                continue
            
            print(f"\n▶ Run {run_idx}/{args.runs} | Model: {model_short_name}")
            run_evaluation_loop(
                responses_path=Path(args.responses_file),
                checklist_path=Path(args.checklist_file),
                output_path=run_file,
                model_name=model,
                workers=args.workers,
                resume=True
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation.")
    
    parser.add_argument("--responses-file", type=str, required=True, help="Path to model responses JSON")
    parser.add_argument("--checklist-file", type=str, required=True, help="Path to instructions/checklist JSON")
    parser.add_argument("--output-dir", type=str, default="output/evals", help="Base output directory")
    
    parser.add_argument("--models", nargs="+", help="List of model keys to evaluate (e.g. 'gpt-4')")
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs per model")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Overwrite existing runs")

    args = parser.parse_args()
    main(args)
import json
import csv
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, List

# =============================================================================
# 1. Helpers
# =============================================================================

def is_instruction_item(item: Dict) -> bool:
    """
    Check if item is an instruction based on [I prefix.
    Used to filter EVALUATION data to match the calibration mode.
    """
    req = item.get("requirement", "")
    return isinstance(req, str) and req.startswith("[I")

# =============================================================================
# 2. Calibration Profile Loading
# =============================================================================

def load_all_judges_from_json(calibration_json_path: Path, subset_key: str) -> Dict[str, Dict]:
    """
    Load judge profiles based on the JSON structure:
    {
       "judge_name": {
           "instr": { "q0": ..., "q1": ... },
           "total": { "q0": ..., "q1": ... }
       }
    }
    
    Parameters
    ----------
    calibration_json_path : Path
        Path to judge_profiles.json
    subset_key : str
        'instr_only' or 'full' (internal script keys)
    """
    if not calibration_json_path.exists():
        print(f"❌ Calibration file not found: {calibration_json_path}")
        return {}

    try:
        with open(calibration_json_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return {}

    # Mapping between script subset keys and JSON keys
    # script: "instr_only" -> json: "instr"
    # script: "full"       -> json: "total"
    json_target_key = "instr" if subset_key == "instr_only" else "total"

    print(f"⚖️  Loading calibration profiles for mode: '{subset_key}' (JSON key: '{json_target_key}')...")
    
    profiles = {}

    # Iterates over each judge in the JSON file
    for judge_name, judge_content in full_data.items():
        
        # Checks if the judge has data for the desired subset (instr or total)
        if json_target_key in judge_content:
            metrics = judge_content[json_target_key]
            
            # If metrics is None (can happen if the calculation failed in the previous step), skip
            if not metrics:
                continue

            profiles[judge_name] = {
                "metrics": {
                    "q0": float(metrics["q0"]),
                    "q1": float(metrics["q1"]),
                    "m0": int(metrics["m0"]),
                    "m1": int(metrics["m1"])
                }
            }

    return profiles

# =============================================================================
# 3. Bootstrapping Logic
# =============================================================================

def calculate_bootstrapped_metrics(instruction_results: List[List[bool]],
                                   b_iterations: int,
                                   rng: np.random.Generator,
                                   sample_size: int = 150):
    """
    Performs subsampling bootstrap with:
    - item-level calibration (q0, q1)
    - instruction-level normalization
    - instruction-level aggregation
    (Model A: item-calibrated, instruction-normalized)
    """
    boot_thetas = []
    
    N = len(instruction_results)
    if N == 0:
        return 0.0, 0.0, 0.0
    
    # Enforce strict cardinality control
    if N < sample_size:
        return np.nan, np.nan, np.nan

    for _ in range(b_iterations):
        # Subsample instructions (clusters)
        resampled_indices = rng.choice(N, size=sample_size, replace=False)

        instr_scores = []

        for idx in resampled_indices:
            checks = instruction_results[idx]

            # --- Item-level calibration ---
            calibrated_items = []
            for c in checks:
                y = 1.0 if c is True else 0.0
                calibrated_items.append(y)

            if calibrated_items:
                # --- Instruction-level normalization ---
                instr_score = float(np.mean(calibrated_items))
                instr_scores.append(instr_score)

        if not instr_scores:
            continue

        # --- Model-level aggregation ---
        theta_boot = float(np.mean(instr_scores))
        boot_thetas.append(theta_boot)

    if not boot_thetas:
        return 0.0, 0.0, 0.0

    return (
        float(np.mean(boot_thetas)),            # Mean Theta
        float(np.percentile(boot_thetas, 2.5)), # CI Lower
        float(np.percentile(boot_thetas, 97.5)) # CI Upper
    )



# =============================================================================
# 4. Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-eval-dir", type=str, default="output/evals")
    parser.add_argument("--calibration-json", type=str, default="output/tables/judge_profiles.json")
    parser.add_argument("--output-csv", type=str, default="output/tables/calibrated_scores_sampled.csv")
    parser.add_argument("--bootstrap-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    BOOTSTRAP_RNG = np.random.default_rng(args.seed)

    results_list = []
    
    # Modes mapping: (script_key, is_instruction_only_flag)
    modes = [("full", False), ("instr_only", True)]

    for mode_key, is_instr_only in modes:
        print(f"\n🚀 Processing Subset: {mode_key.upper()}")

        # 1. Load calibration profiles specific to this mode from JSON
        judge_data = load_all_judges_from_json(Path(args.calibration_json), subset_key=mode_key)
        
        if not judge_data:
            print(f"⚠️ Skipping {mode_key} (no valid judge data found in JSON)")
            continue
            
        # 3. Iterate Models
        base_path = Path(args.base_eval_dir)
        if not base_path.exists():
            print("❌ Base eval directory not found.")
            return

        for model_dir in sorted(base_path.iterdir(), key=lambda p: p.name):
            if not model_dir.is_dir(): continue
            
            runs_dir = model_dir / "runs"
            if not runs_dir.exists(): continue

            for run_folder in sorted(runs_dir.iterdir(), key=lambda p: p.name):
                # Extract judge name
                judge_name = run_folder.name[4:] if run_folder.name.startswith("run_") else run_folder.name
                
                # Check if we have calibration data for this judge
                if judge_name not in judge_data: continue

                # Find results file
                valid_files = sorted(
                    [f for f in run_folder.glob("*_results.json") if "_checkpoint" not in f.name],
                    key=lambda p: p.name
                )
                if not valid_files: continue

                # Load Evaluation Data
                try:
                    data = json.loads(valid_files[0].read_text(encoding='utf-8'))
                except Exception:
                    continue

                results = data.get("results", data)
                
                # Collect checks per instruction (cluster)
                instr_level_data = []
                total_items = 0
                
                for _, content in results.items():
                    checklist = content.get("checklist", {})
                    items = checklist.get("items", []) if isinstance(checklist, dict) else []
                    
                    # --- FILTERING STEP ---
                    valid_checks = []
                    for it in items:
                        check = it.get("check")
                        if check not in [True, False]:
                            continue
                        
                        # Apply Filter: if mode is instr_only, skip non-instruction items
                        if is_instr_only and not is_instruction_item(it):
                            continue
                            
                        valid_checks.append(check)
                    
                    if valid_checks:
                        instr_level_data.append(valid_checks)
                        total_items += len(valid_checks)

                if not instr_level_data: continue
                
                # --- Bootstrap: Individual Judge ---
                theta, ci_l, ci_u = calculate_bootstrapped_metrics(
                    instr_level_data,
                    args.bootstrap_runs,
                    BOOTSTRAP_RNG,
                    sample_size=150,
                )

                results_list.append({
                    "model": model_dir.name,
                    "judge": judge_name,
                    "subset": mode_key,
                    "n_instructions": len(instr_level_data),
                    "n_items": total_items,
                    "theta_calibrated": round(theta, 4),
                    "ci_lower": round(ci_l, 4),
                    "ci_upper": round(ci_u, 4)
                })

    # Save Results
    if results_list:
        results_list.sort(key=lambda x: (x['subset'], x['theta_calibrated']), reverse=True)
        
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\n✅ Bootstrapped Ranking saved at: {out_path}")
    else:
        print("\n⚠️ No results computed.")

if __name__ == "__main__":
    main()
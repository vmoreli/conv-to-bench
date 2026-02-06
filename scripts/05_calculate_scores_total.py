import json
import csv
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Set, Iterable

# =============================================================================
# Utils
# =============================================================================

def get_item_fingerprint(item: Dict, model_context: Optional[str] = None, manual_key: Optional[str] = None):
    o_key = manual_key or item.get("original_key")
    req = item.get("requirement")
    model = item.get("modelo_avaliado") or model_context
    if not o_key or not req or not model:
        return None
    return (str(o_key), str(req), str(model))


def extract_judge_name_from_folder(folder_name: str) -> str:
    return folder_name[4:] if folder_name.startswith("run_") else folder_name


def is_instruction_item(item: Dict) -> bool:
    req = item.get("requirement", "")
    return isinstance(req, str) and req.startswith("[I")


def safe_iter_results(results_obj) -> Iterable[Tuple[str, Dict]]:
    if isinstance(results_obj, dict):
        return results_obj.items()
    elif isinstance(results_obj, list):
        return ((str(i), x) for i, x in enumerate(results_obj))
    else:
        return []

# =============================================================================
# Fingerprints (anti-leakage)
# =============================================================================

def get_calibration_fingerprints(judge_dir: Path, instruction_only: bool = False) -> Set[Tuple]:
    calibration_fingerprints: Set[Tuple] = set()
    json_files = list(judge_dir.glob("*.json"))
    if not json_files:
        return set()

    for file_path in json_files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            samples = data.get("samples", data)
            all_items = []

            if isinstance(samples, dict):
                for key in ["true", "false"]:
                    all_items.extend(samples.get(key, []))
            elif isinstance(samples, list):
                all_items = samples

            for item in all_items:
                if instruction_only and not is_instruction_item(item):
                    continue
                fingerprint = get_item_fingerprint(item)
                if fingerprint and item.get("check_gt") is not None:
                    calibration_fingerprints.add(fingerprint)
        except Exception:
            pass

    return calibration_fingerprints

# =============================================================================
# Cluster Bootstrap Estimator
# =============================================================================

def calculate_bootstrap_instruction_ci(
    run_file: Path,
    exclude_set: Set[Tuple],
    model_name: str,
    rng: np.random.Generator,
    B: int = 1000,
    instruction_only: bool = False
):
    """
    Hierarchical estimator:
    - cluster = instruction
    - item calibration
    - instruction normalization
    - model aggregation
    - cluster bootstrap CI
    """

    try:
        data = json.loads(run_file.read_text(encoding="utf-8"))
    except Exception:
        return 0.0, 0.0, 0.0, 0, 0

    results = data.get("results", data)

    # ---------- build instruction clusters ----------
    instruction_clusters = []  # List[List[float]] calibrated items
    n_items = 0

    for key, content in safe_iter_results(results):
        checklist = content.get("checklist", {})
        items = checklist.get("items", []) if isinstance(checklist, dict) else []

        calibrated_items = []

        for item in items:
            if instruction_only and not is_instruction_item(item):
                continue

            fingerprint = get_item_fingerprint(item, model_context=model_name, manual_key=key)
            if fingerprint is None or fingerprint in exclude_set:
                continue

            check_val = item.get("check")
            if check_val not in [True, False]:
                continue

            y = 1.0 if check_val else 0.0

            calibrated_items.append(y)
            n_items += 1

        if calibrated_items:
            instruction_clusters.append(calibrated_items)

    N_instr = len(instruction_clusters)
    if N_instr == 0:
        return 0.0, 0.0, 0.0, 0, 0

    # ---------- bootstrap ----------
    boot_thetas = []

    for _ in range(B):
        idxs = rng.choice(N_instr, size=N_instr, replace=True)
        instr_scores = []

        for i in idxs:
            cluster = instruction_clusters[i]
            instr_scores.append(float(np.mean(cluster)))

        theta_b = float(np.mean(instr_scores))
        boot_thetas.append(theta_b)

    theta = float(np.mean(boot_thetas))
    ci_l = float(np.percentile(boot_thetas, 2.5))
    ci_u = float(np.percentile(boot_thetas, 97.5))

    return theta, ci_l, ci_u, N_instr, n_items

# =============================================================================
# Main
# =============================================================================

def main(args):
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    output_file = Path(args.output_csv)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.judge_profiles_json, "r", encoding="utf-8") as f:
        loaded_profiles = json.load(f)

    ablation_modes = {
        "instr_only": {"is_instr": True, "json_key": "instr"},
        "full":       {"is_instr": False, "json_key": "total"}
    }

    annotations_base = Path(args.annotations_base)
    base_eval = Path(args.base_eval_dir)

    results_list = []

    for subset_label, config in ablation_modes.items():
        is_instr = config["is_instr"]
        json_key = config["json_key"]

        current_run_judges = {}

        for judge_dir in annotations_base.iterdir():
            if not judge_dir.is_dir():
                continue
            judge_name = judge_dir.name
            if judge_name not in loaded_profiles:
                continue
            profile_metrics = loaded_profiles[judge_name].get(json_key)
            if not profile_metrics:
                continue

            fprints = get_calibration_fingerprints(judge_dir, instruction_only=is_instr)

            current_run_judges[judge_name] = {
                "metrics": profile_metrics,
                "fingerprints": fprints
            }

        for model_dir in base_eval.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            runs_dir = model_dir / "runs"
            if not runs_dir.exists():
                continue

            for run_folder in runs_dir.iterdir():
                if not run_folder.is_dir():
                    continue

                judge_name = extract_judge_name_from_folder(run_folder.name)
                judge_data = current_run_judges.get(judge_name)
                if not judge_data:
                    continue

                valid_files = [f for f in run_folder.glob("*_results.json") if "_checkpoint" not in f.name]
                if not valid_files:
                    continue

                theta, ci_l, ci_u, n_instr, n_items = calculate_bootstrap_instruction_ci(
                    valid_files[0],
                    judge_data["fingerprints"],
                    model_name,
                    rng,
                    B=args.bootstrap_runs,
                    instruction_only=is_instr
                )

                if n_instr == 0:
                    continue

                results_list.append({
                    "subset": subset_label,
                    "model_evaluated": model_name,
                    "judge_name": judge_name,
                    "n_instructions": n_instr,
                    "n_items": n_items,
                    "theta_calibrated": round(theta, 4),
                    "ci_lower": round(ci_l, 4),
                    "ci_upper": round(ci_u, 4),
                    "estimator": "item-calibrated / instruction-normalized",
                    "ci_method": "cluster_bootstrap_instruction"
                })

    # ---------- Export ----------
    if results_list:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
            writer.writeheader()
            writer.writerows(results_list)

        print(f"\n✅ Paper-ready CSV generated: {output_file}")
    else:
        print("⚠️ No results generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-eval-dir", type=str, default="output/evals")
    parser.add_argument("--annotations-base", type=str, default="data/annotations")
    parser.add_argument("--judge-profiles-json", type=str, default="output/tables/judge_profiles.json")
    parser.add_argument("--output-csv", type=str, default="output/tables/calibrated_scores.csv")
    parser.add_argument("--bootstrap-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())

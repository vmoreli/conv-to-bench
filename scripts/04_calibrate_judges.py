import json
import argparse
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import cohen_kappa_score, f1_score


def is_instruction_item(item: Dict) -> bool:
    """Identifies instruction items ([I])."""
    req = item.get("requirement", "")
    return isinstance(req, str) and req.startswith("[I")


def calculate_judge_metrics(judge_dir: Path) -> Dict:
    """
    Calculates:
    - q0 (specificity)
    - q1 (sensitivity)
    - Cohen's Kappa
    - F1-score per class
    - Macro-F1

    for categories:
    'instr' (instructions only) and 'total' (all items regardless of type).
    """

    categories = {
        "instr": {"gt": [], "pred": []},
        "non_instr": {"gt": [], "pred": []},
        "total": {"gt": [], "pred": []}
    }

    
    for file_path in judge_dir.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            samples = data.get("samples", data)
            items: List[Dict] = []
            if isinstance(samples, dict):
                for k in ["true", "false"]:
                    items.extend(samples.get(k, []))
            elif isinstance(samples, list):
                items = samples

            for item in items:
                gt = item.get("check_gt")   # ground truth
                check = item.get("check")  # judge decision
                
                if gt is not None and check is not None:
                    # TOTAL
                    categories["total"]["gt"].append(bool(gt))
                    categories["total"]["pred"].append(bool(check))

                    # SPLIT instr vs non-instr
                    if is_instruction_item(item):
                        categories["instr"]["gt"].append(bool(gt))
                        categories["instr"]["pred"].append(bool(check))
                    else:
                        categories["non_instr"]["gt"].append(bool(gt))
                        categories["non_instr"]["pred"].append(bool(check))

        except Exception as e:
            print(f"  ⚠️ Error in {file_path.name}: {e}")

    def compute_stats(gt: List[bool], pred: List[bool]):
        if len(gt) == 0:
            return None

        # Confusion components
        tp = sum(1 for g, p in zip(gt, pred) if g is True and p is True)
        tn = sum(1 for g, p in zip(gt, pred) if g is False and p is False)
        fp = sum(1 for g, p in zip(gt, pred) if g is False and p is True)
        fn = sum(1 for g, p in zip(gt, pred) if g is True and p is False)

        # Accuracy (%)
        correct = sum(1 for g, p in zip(gt, pred) if g == p)
        accuracy = correct / len(gt) if len(gt) > 0 else None

        m1 = sum(gt)              # gt_true
        m0 = len(gt) - m1         # gt_false

        # Sensitivity (q1)
        q1 = tp / m1 if m1 > 0 else None
        # Specificity (q0)
        q0 = tn / m0 if m0 > 0 else None

        # Cohen's Kappa
        kappa = cohen_kappa_score(gt, pred)

        # F1 per class
        # class False = 0, True = 1
        f1_false = f1_score(gt, pred, pos_label=False)
        f1_true = f1_score(gt, pred, pos_label=True)

        # Macro-F1
        macro_f1 = (f1_false + f1_true) / 2

        return {
            "q0": q0,
            "q1": q1,
            "kappa": kappa,
            "f1_false": f1_false,
            "f1_true": f1_true,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "confusion": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            },
            "m0": m0,
            "m1": m1,
            "n": len(gt)
        }


    return {
        "instr": compute_stats(categories["instr"]["gt"], categories["instr"]["pred"]),
        "non_instr": compute_stats(categories["non_instr"]["gt"], categories["non_instr"]["pred"]),
        "total": compute_stats(categories["total"]["gt"], categories["total"]["pred"])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/annotations")
    parser.add_argument("--output-json", type=str, default="output/tables/judge_profiles.json")
    args = parser.parse_args()

    profiles = {}
    input_path = Path(args.input_dir)
    
    # Create output directory if it doesn't exist
    output_file = Path(args.output_json)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    for judge_dir in input_path.iterdir():
        if judge_dir.is_dir():
            print(f"⚙️ Processing metrics: {judge_dir.name}...")
            metrics = calculate_judge_metrics(judge_dir)
            if metrics["total"]:
                profiles[judge_dir.name] = metrics

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Calibration JSON generated: {args.output_json}")


if __name__ == "__main__":
    main()

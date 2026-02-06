import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

"""
Compare calibrated metrics against reference benchmarks.

This script loads calibrated judge scores (total and sampled/bootstrapped),
computes correlation metrics (Spearman, Kendall) against reference benchmarks
(BigCodeBench Hard and HumanEval win rates), and outputs a comparison table.
"""

# =============================================================================
# 1. REFERENCE DATA
# =============================================================================
REFERENCE_DATA = [
    {
        "model": "gpt-4.1-nano-2025-04-14",
        "bigcodebench": {
            "hard": {
                "complete": 31.8,
                "instruct": 28.4,
                "average": 30.1
            },
            "full": {
                "complete": np.nan,
                "instruct": np.nan,
                "average": np.nan
            }
        },
        "human_eval": {
            "winrate": np.nan,
            "python": np.nan
        }
    },
    {
        "model": "gemini-2.0-flash-001",
        "bigcodebench": {
            "hard": {
                "complete": 33.8,
                "instruct": 23.6,
                "average": 28.7
            },
            "full": {
                "complete": np.nan,
                "instruct": np.nan,
                "average": np.nan
            }
        },
        "human_eval": {
            "winrate": np.nan,
            "python": np.nan
        }
    },
    {
        "model": "Qwen2.5-Coder-7B-Instruct",
        "bigcodebench": {
            "hard": {
                "complete": 20.3,
                "instruct": 20.3,
                "average": 20.3
            },
            "full": {
                "complete": 48.8,
                "instruct": 40.4,
                "average": 44.6
            }
        },
        "human_eval": {
            "winrate": np.nan,
            "python": np.nan
        }
    },
    {
        "model": "CodeQwen1.5-7B-Chat",
        "bigcodebench": {
            "hard": {
                "complete": 15.5,
                "instruct": 18.9,
                "average": 17.2
            },
            "full": {
                "complete": 43.6,
                "instruct": 39.6,
                "average": 41.6
            }
        },
        "human_eval": {
            "winrate": 55.67,
            "python": 87.2
        }
    },
    {
        "model": "deepseek-coder-6.7b-instruct",
        "bigcodebench": {
            "hard": {
                "complete": 15.5,
                "instruct": 10.1,
                "average": 12.8
            },
            "full": {
                "complete": 43.8,
                "instruct": 35.5,
                "average": 39.6
            }
        },
        "human_eval": {
            "winrate": 50.58,
            "python": 80.22
        }
    },
    {
        "model": "codegemma-7b-it",
        "bigcodebench": {
            "hard": {
                "complete": 13.5,
                "instruct": 7.4,
                "average": 10.4
            },
            "full": {
                "complete": 39.3,
                "instruct": 32.3,
                "average": 35.8
            }
        },
        "human_eval": {
            "winrate": 27.0,
            "python": 42.74
        }
    },
    {
        "model": "CodeLlama-13b-Instruct-hf",
        "bigcodebench": {
            "hard": {
                "complete": 6.8,
                "instruct": 9.5,
                "average": 8.2
            },
            "full": {
                "complete": 31.7,
                "instruct": 28.5,
                "average": 30.1
            }
        },
        "human_eval": {
            "winrate": 34.35,
            "python": 50.6
        }
    },
    {
        "model": "CodeLlama-7b-Instruct-hf",
        "bigcodebench": {
            "hard": {
                "complete": 4.1,
                "instruct": 3.4,
                "average": 3.8
            },
            "full": {
                "complete": 25.7,
                "instruct": 21.9,
                "average": 23.8
            }
        },
        "human_eval": {
            "winrate": 28.46,
            "python": 45.65
        }
    }
]

# =============================================================================
# 2. UTILITIES
# =============================================================================

def flatten_reference_data(ref_data: list) -> pd.DataFrame:
    """
    Flatten hierarchical REFERENCE_DATA into a tabular format.
    """
    df = pd.json_normalize(ref_data, sep=".")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df["model"] = df["model"].apply(normalize_model_name)
    return df

def normalize_model_name(name: str) -> str:
    """Extract the last component of a model name (e.g., 'org/model' -> 'model')."""
    if not isinstance(name, str): return str(name)
    return name.split('/')[-1].strip()

def load_arenahard_json(json_path: Path, judge_label: str) -> pd.DataFrame:
    """Load ArenaHard JSON, handling both nested 'economica' or direct format.
    
    Reads ArenaHard results and extracts score and confidence interval,
    returning a DataFrame compatible with the comparison pipeline.
    """
    if not json_path.exists():
        print(f"⚠️  File not found: {json_path}")
        return pd.DataFrame()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    models_dict = data.get("models", {})
    
    for model_name, content in models_dict.items():
        # Flexible lookup: check 'economica' nested field or root of model object
        target = content.get("economica", content)
        
        if "score" in target and "confidence_interval_score" in target:
            rows.append({
                "model_evaluated": normalize_model_name(model_name),
                "judge_name": judge_label,
                "theta_calibrated": target["score"],
                "ci_lower": target["confidence_interval_score"][0],
                "ci_upper": target["confidence_interval_score"][1],
                "subset": ""   
            })
            
    return pd.DataFrame(rows)


# =============================================================================
# 3. METRICS MODULE
# =============================================================================

def calculate_validation_metrics(df_bench: pd.DataFrame, method_label: str) -> pd.DataFrame:
    """Compute correlation and agreement metrics for a set of calibrated judge scores.
    
    Compares judge rankings against reference benchmarks using Spearman/Kendall
    correlations and pairwise agreement rates for models with separable CIs.
    """
    if df_bench.empty: 
        return pd.DataFrame()
    
    # Ensure compatibility with older CSVs that may not have 'subset' column
    if 'subset' not in df_bench.columns:
        df_bench['subset'] = 'full'   # Default to 'full' if not present

    df_ref_raw = flatten_reference_data(REFERENCE_DATA)

    df_bench = df_bench.rename(columns={'model': 'model_evaluated', 'judge': 'judge_name'})
    df_bench['model_evaluated'] = df_bench['model_evaluated'].apply(normalize_model_name)

    # Auto-detect all benchmark columns
    ref_columns = [
        c for c in df_ref_raw.columns
        if c.startswith("bigcodebench_") or c.startswith("human_eval_")
    ]

    summary_results = []

    # Iterate over judge and subset combinations
    for (judge, subset), df_js in df_bench.groupby(['judge_name', 'subset']):
        if judge == 'deepseek':
            continue

        for ref_col in ref_columns:
            valid_ref = df_ref_raw.dropna(subset=[ref_col])
            common_models = sorted(list(set(valid_ref['model']).intersection(set(df_js['model_evaluated']))))

            if len(common_models) < 3:
                continue

            df_ref_sub = valid_ref[valid_ref['model'].isin(common_models)].sort_values('model')
            df_j_sub = df_js[df_js['model_evaluated'].isin(common_models)].sort_values('model_evaluated')

            spearman_r, spearman_p = spearmanr(df_ref_sub[ref_col], df_j_sub['theta_calibrated'])
            kendall_r, kendall_p = kendalltau(df_ref_sub[ref_col], df_j_sub['theta_calibrated'])

            # Pairwise agreement computation
            ref_vals = df_ref_sub[ref_col].values
            models = df_ref_sub['model'].values
            j_thetas = df_j_sub['theta_calibrated'].values
            j_low = df_j_sub['ci_lower'].values
            j_high = df_j_sub['ci_upper'].values

            total_pairs, separable, agreements, disagreements = 0, 0, 0, 0
            for i in range(len(common_models)):
                for k in range(i + 1, len(common_models)):
                    total_pairs += 1
                    is_sep = (j_low[i] > j_high[k]) or (j_low[k] > j_high[i])
                    if is_sep:
                        separable += 1
                        ref_winner = models[i] if ref_vals[i] > ref_vals[k] else models[k]
                        j_winner = models[i] if j_thetas[i] > j_thetas[k] else models[k]
                        if ref_winner == j_winner:
                            agreements += 1
                        else:
                            disagreements += 1

            summary_results.append({
                "method": method_label,
                "judge": judge,
                "subset": subset,   
                "ref_benchmark": ref_col,
                "spearman_r": round(spearman_r, 4),
                "spearman_p": round(spearman_p, 4),
                "kendall_r": round(kendall_r, 4),
                "kendall_p": round(kendall_p, 4),
                "conf_agreement": round((agreements - disagreements) / total_pairs, 4) if total_pairs > 0 else 0,
                "sep_rate": f"{round((separable/total_pairs)*100, 1)}%",
                "n_models": len(common_models)
            })

    return pd.DataFrame(summary_results)

# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Calibrated scores computed on full evaluation dataset
    parser.add_argument("--total-csv", type=str, default="output/tables/calibrated_scores.csv")

    # Calibrated scores computed on sampled/bootstrapped data
    parser.add_argument("--sampled-csv", type=str, default="output/tables/calibrated_scores_sampled.csv")

    # ArenaHard reference files (for external validation)
    parser.add_argument("--arena-gemini", type=str, default="data/arenahard/analysis_results_gemini.json")
    parser.add_argument("--arena-gpt", type=str, default="data/arenahard/analysis_results_gpt.json")

    # Output consolidated comparison table
    parser.add_argument("--output", type=str, default="output/tables/comparison_table.csv")

    args = parser.parse_args()

    all_dfs = []

    # Load and process evaluation results from various sources
    if Path(args.total_csv).exists():
        all_dfs.append(calculate_validation_metrics(pd.read_csv(args.total_csv), "01_Total_Data"))
    
    if Path(args.sampled_csv).exists():
        all_dfs.append(calculate_validation_metrics(pd.read_csv(args.sampled_csv), "02_Sampled_150_BS"))

    # ArenaHard Gemini (nested format with 'economica')
    df_ag = load_arenahard_json(Path(args.arena_gemini), "gemini-2.5-flash")
    if not df_ag.empty:
        all_dfs.append(calculate_validation_metrics(df_ag, "03_ArenaHard_economica"))

    # ArenaHard GPT (direct format)
    df_ap = load_arenahard_json(Path(args.arena_gpt), "gpt-5-mini")
    if not df_ap.empty:
        all_dfs.append(calculate_validation_metrics(df_ap, "03_ArenaHard_economica"))

    if not all_dfs:
        print("❌ No data found."); return

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sort_values(
        by=["ref_benchmark", "judge", "subset", "method"],
        ascending=[True, True, True, True]   # method in explicit ascending order
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    
    print(f"✅ Success! Consolidated report saved to: {args.output}")

if __name__ == "__main__":
    main()
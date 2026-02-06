# scripts/01_topic_clustering.py
import os
import json
import argparse
import time
import re
import ast
import numpy as np
import pandas as pd
import plotly.io as pio
        
from datasets import load_from_disk

# Project imports
from src.data.extraction import extract_conversations_from_dataset
from src.clustering.embeddings import generate_embeddings
from src.clustering.model import train_bertopic

# --- CODE KEYWORDS CONFIGURATION ---
CODE_KEYWORDS = [
    "python", "javascript", "typescript", "java", "c", "c++", "cpp", "csharp",
    "go", "rust", "ruby", "kotlin", "swift", "scala", "perl", "php", "bash",
    "shell", "haskell", "elixir", "dart", "r", "lua", "julia", "sql",
    "async", "await", "lambda", "yield", "match", "case", "break", "continue",
    "elif", "except", "finally", "raise", "global", "nonlocal", "del", "pass",
    "typedef", "struct", "interface", "implements", "override", "extends",
    "enum", "trait", "impl", "defer", "package", "import", "export", "namespace",
    "sizeof", "volatile", "static", "inline", "fn", "mut", "const", "var", "let",
    "closure", "decorator", "pointer", "reference", "recursion", "stack", "heap",
    "bytecode", "casting", "serialization", "deserialization", "memoization",
    "inheritance", "encapsulation", "polymorphism", "constructor", "destructor",
    "git", "github", "gitlab", "commit", "push", "pull", "merge", "branch",
    "rebase", "docker", "kubernetes", "makefile", "cmake", "venv", "pip",
    "npm", "yarn", "conda", "virtualenv", "linter", "formatter", "eslint",
    "prettier", "ci", "cd", "pipeline", "build", "deploy", "debugger",
    "react", "vue", "angular", "svelte", "express", "django", "flask",
    "fastapi", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "scikit", "tensorflow", "keras", "pytorch", "torch", "beautifulsoup",
    "compilation", "compiler", "interpreter", "runtime", "syntax", "tokenizer",
    "parser", "segfault", "segmentation fault", "repl", "shell script"
]

def _check_for_keywords_in_cell(cell_content, pattern):
    """Checks if keywords are present in the cell content."""
    if pd.isna(cell_content):
        return False
    
    # If it's a string that looks like a list/tuple (common in BERTopic output), try to parse it
    if isinstance(cell_content, str) and (cell_content.startswith('[') or cell_content.startswith('(')):
        try:
            evaluated_content = ast.literal_eval(cell_content)
            if isinstance(evaluated_content, (list, tuple)):
                return any(isinstance(item, str) and re.search(pattern, item, re.IGNORECASE) for item in evaluated_content)
        except (ValueError, SyntaxError):
            pass
    
    # Treat as a regular string
    return bool(re.search(pattern, str(cell_content), re.IGNORECASE))

def filter_code_topics(df, keywords):
    """Filters the DataFrame of topics based on the keywords."""
    cols_to_check = ['Representation', 'POS', 'KeyBERT']
    existing_cols = [col for col in cols_to_check if col in df.columns]
    
    if not existing_cols:
        return pd.DataFrame()

    pattern = r'\b(?:' + '|'.join([re.escape(word) for word in keywords]) + r')\b'
    
    mask = df[existing_cols].apply(
        lambda col: col.apply(lambda x: _check_for_keywords_in_cell(x, pattern))
    ).any(axis=1)
    
    return df[mask]

def main(args):
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = args.dataset_name or os.path.basename(os.path.abspath(args.dataset_dir))

    # Define Output Paths
    paths = {
        "conversations": os.path.join(args.output_dir, f"processed_conversations_{dataset_name}.json"),
        "embeddings": os.path.join(args.output_dir, f"embeddings_{dataset_name}.npy"),
        "model_dir": os.path.join(args.output_dir, f"bertopic_model_{dataset_name}"),
        "topics_csv": os.path.join(args.output_dir, f"topics_info_{dataset_name}.csv"),
        "code_topics_csv": os.path.join(args.output_dir, f"code_topics_info_{dataset_name}.csv"),
        "visualization_html": os.path.join(args.output_dir, f"topics_viz_{dataset_name}.html"),
        "topic_mapping_json": os.path.join(args.output_dir, f"topic_assignments_{dataset_name}.json")
    }

    # --- STEP 1: LOAD DATASET ---
    dataset_path = os.path.join(args.dataset_dir, args.split) if args.split else args.dataset_dir
    print(f"📂 Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    print("🌍 Filtering for English content...")
    dataset = dataset.filter(lambda x: (x.get("language") == "English") or (x.get("lang") == "en"))
    print(f"✅ {len(dataset)} items remaining after language filtering.")

    # --- STEP 2: PROCESS CONVERSATIONS ---
    if args.skip_existing and os.path.exists(paths["conversations"]):
        print("📄 Found existing processed conversations. Loading...")
        with open(paths["conversations"], "r") as f:
            metadata = json.load(f)
        conversations = [item["processed_text"] for item in metadata]
    else:
        print("⚙️ Processing dataset to extract user messages...")
        start_time = time.time()
        conversations, metadata = extract_conversations_from_dataset(
            dataset, 
            first_n=args.first_n, 
            min_length=args.min_length
        )
        print(f"✅ Extracted {len(conversations)} messages in {time.time() - start_time:.2f} seconds.")
        
        with open(paths["conversations"], "w") as f:
            json.dump(metadata, f, indent=2)

    # --- STEP 3: GENERATE EMBEDDINGS ---
    if args.skip_existing and os.path.exists(paths["embeddings"]):
        print("📦 Found existing embeddings. Loading...")
        embeddings = np.load(paths["embeddings"])
    else:
        start_time = time.time()
        embeddings = generate_embeddings(conversations, model_name=args.embed_model)
        np.save(paths["embeddings"], embeddings)
        print(f"✅ Embeddings generated in {time.time() - start_time:.2f} seconds.")

    # --- STEP 4: TOPIC MODELING (BERTopic) ---
    if args.skip_existing and os.path.exists(paths["model_dir"]):
        print("⚠️ Model already exists. Skipping training.")
        # Optional: load the model here if you need to filter after skip
    else:
        print("🚀 Training BERTopic model...")
        start_time = time.time()
        
        topic_model, topics, _ = train_bertopic(
            conversations, 
            embeddings, 
            embed_model_name=args.embed_model, 
            min_topic_size=args.min_topic_size
        )
        
        print(f"✅ Clustering completed in {time.time() - start_time:.2f} seconds.")

        # Outlier Reduction
        print("🧹 Reducing outliers...")
        new_topics = topic_model.reduce_outliers(conversations, topics)
        
        # Save Topic Assignments
        topic_assignment = {
            metadata[i]["conversation_id"]: new_topics[i] 
            for i in range(len(metadata))
        }
        with open(paths["topic_mapping_json"], "w") as f:
            json.dump(topic_assignment, f, default=str, indent=2)

        # Generate Visualization
        print("📊 Generating interactive visualization...")
        fig = topic_model.visualize_topics()
        pio.write_html(fig, file=paths["visualization_html"], auto_open=False)
        
        # Save Model and Info
        print("📁 Saving model artifacts...")
        topic_model.save(paths["model_dir"], serialization="pytorch", save_ctfidf=True)
        
        df_topics = topic_model.get_topic_info()
        df_topics.to_csv(paths["topics_csv"], index=False)

        # --- STEP 5: FILTER CODE CLUSTERS ---
        if args.filter_code:
            print("💻 Identifying code-related clusters...")
            df_code = filter_code_topics(df_topics, CODE_KEYWORDS)
            
            if not df_code.empty:
                df_code.to_csv(paths["code_topics_csv"], index=False)
                print(f"✅ Found {len(df_code)} code-related topics. Saved to: {paths['code_topics_csv']}")
            else:
                print("ℹ️ No code-related topics identified with current keywords.")

    print(f"✅ Pipeline finished! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run topic clustering pipeline.")
    
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the HuggingFace dataset directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--output-dir", type=str, default="topic_model_st", help="Directory to save results")
    parser.add_argument("--dataset-name", type=str, default=None, help="Short name for output files")
    parser.add_argument("--first-n", type=int, default=None, help="Number of examples to process")
    parser.add_argument("--min-length", type=int, default=32, help="Minimum character length")
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--min-topic-size", type=int, default=32, help="Minimum size for a topic")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps if output files exist")
    parser.add_argument("--filter-code", action="store_true", help="Identify and export code-related clusters")

    args = parser.parse_args()
    main(args)
import os
import json
import argparse
import time

import numpy as np
from tqdm import tqdm

from datasets import load_from_disk

import spacy.cli
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech
from sentence_transformers import SentenceTransformer

import plotly.io as pio


def extract_first_user_message(conversation):
    """
    Extracts the first message sent by the user in a conversation.
    Assumes that each conversation is a list of dictionaries with keys "role" and "content".
    """
    if not conversation:
        return None

    # Try to get the first message directly, if it's from the user
    first = conversation[0]
    if first.get("role") != "user":
        # If not, search for the first one with role == "user"
        for msg in conversation:
            if msg.get("role") == "user":
                first = msg
                break
        else:
            return None

    content = first.get("content")
    if isinstance(content, list) and content:
        return str(content[0])
    elif isinstance(content, str):
        return content
    else:
        return str(content)


def build_convs_from_dataset(ds, first_n=None, min_len=32, max_len=10_000):
    """
    Iterates through the dataset and extracts the first relevant message (user or first message in a conversation).
    Performs simple preprocessing and stores texts and metadata.
    """
    convs = []
    meta = []

    # Determine the dataset structure type: conversation per line or message per line
    is_conversation_per_line = False
    is_message_per_line = False
    
    if 'openai_moderation' in ds.column_names:
        print(f"📂 Filtering according to OpenAI Moderation.")
        # Filters only examples where NONE of the items in the 'moderation' list were flagged
        ds = ds.filter(lambda x: all(not m["flagged"] for m in x["openai_moderation"]))
    else:
        print(f"⚠️ The 'openai_moderation' column was not found. Filtering was skipped.")

    if len(ds) > 0:
        sample_row = ds[0]
        if "conversation" in sample_row and isinstance(sample_row["conversation"], list):
            is_conversation_per_line = True
        # Prioritizes 'text' for message per line, but still checks 'content' as fallback
        elif "parent_id" in sample_row and ("text" in sample_row or "content" in sample_row):
            is_message_per_line = True
        else:
            print("⚠️ Warning: Could not determine the dataset format (conversation per line or message per line).")
            print("⚠️ Attempting to process as 'conversation per line' by default, which may fail.")
            is_conversation_per_line = True # Default fallback

    it = ds if first_n is None else ds.select(range(min(first_n, len(ds))))

    if is_message_per_line:
        print("⚙️ Detected format: Message per line. Filtering by null 'parent_id'...")
        # Filters to get only the first messages (where parent_id is null/None)
        it = it.filter(lambda x: x.get("parent_id") is None)
        print(f"✅ {len(it)} first messages identified.")

    for row_idx, row in enumerate(tqdm(it, desc="🔍 Extracting initial messages")):
        conv_content = None

        if is_conversation_per_line:
            conv_id = row.get("conversation_id") or row.get("conversation_hash") or f"conv_idx_{row_idx}"
            conv_content = extract_first_user_message(row["conversation"])
        elif is_message_per_line:
            conv_id = row.get("message_tree_id")
            if "text" in row:
                conv_content = row["text"]
            elif "content" in row:
                conv_content = row["content"]
            
            if isinstance(conv_content, list) and conv_content:
                conv_content = str(conv_content[0])
            elif not isinstance(conv_content, str):
                conv_content = str(conv_content)

        if not conv_content:
            continue

        conv_content = conv_content.replace("<|endoftext|>", "<| endoftext |>")
        if len(conv_content) <= min_len:
            continue
        conv_content = conv_content[:max_len]

        convs.append(conv_content)
        meta.append({"conversation_id": conv_id, "post_process_conv": conv_content})

    return convs, meta


def ensure_pos_model():
    """
    Ensures that the spaCy model for grammatical analysis is available.
    """
    try:
        return PartOfSpeech("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        return PartOfSpeech("en_core_web_sm")


def embed_with_st(texts, model_name="all-MiniLM-L6-v2", batch_size=512):
    """
    Generates embeddings using a SentenceTransformers model.
    """
    print(f"📅 Loading model {model_name}...")
    model = SentenceTransformer(model_name)

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔐 Generating embeddings"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)

    return np.vstack(embeddings)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = args.dataset_name or os.path.basename(os.path.abspath(args.dataset_dir))

    # 1. Loading the dataset
    ds_path = os.path.join(args.dataset_dir, args.split) if args.split else args.dataset_dir
    print(f"📂 Loading dataset from: {ds_path}")
    ds = load_from_disk(ds_path)

    # Filter only for conversations in English
    print("🌍 Filtering only English conversations/messages...")

    # Checks if the 'language' key exists and if the value is 'English'
    # OR if the 'lang' key exists and the value is 'en'
    ds = ds.filter(lambda x: (x.get("language") == "English") or (x.get("lang") == "en"))

    print(f"✅ {len(ds)} English conversations/messages remaining")

    convs_path = os.path.join(args.output_dir, f"post_process_convs_{dataset_name}.json")
    embeddings_path = os.path.join(args.output_dir, f"embeddings_{dataset_name}.npy")
    model_dir = os.path.join(args.output_dir, f"model_{dataset_name}")
    topics_csv = os.path.join(args.output_dir, f"topics_{dataset_name}.csv")
    html_path = os.path.join(args.output_dir, f"topics_visualization_{dataset_name}.html")
    topic_json_path = os.path.join(args.output_dir, f"conv_topics_{dataset_name}.json")

    # 2. Processing conversations
    if args.skip_existing and os.path.exists(convs_path):
        print("📄 Processed conversations already exist. Loading...")
        with open(convs_path, "r") as f:
            meta = json.load(f)
        convs = [m["post_process_conv"] for m in meta]
    else:
        print("⚙️ Processing messages...")
        t0 = time.time()
        convs, meta = build_convs_from_dataset(
            ds, first_n=args.first_n, min_len=args.min_len, max_len=args.max_len
        )
        print(f"✅ {len(convs)} messages extracted in {time.time() - t0:.2f} seconds")
        with open(convs_path, "w") as f:
            json.dump(meta, f, indent=2)

    # 3. Generating embeddings
    if args.skip_existing and os.path.exists(embeddings_path):
        print("📦 Embeddings already exist. Loading...")
        embeddings = np.load(embeddings_path)
    else:
        t1 = time.time()
        embeddings = embed_with_st(convs, model_name=args.embed_model)
        np.save(embeddings_path, embeddings)
        print(f"✅ Embeddings generated in {time.time() - t1:.2f} seconds")
        print(f"🕥 Embeddings dimension: {embeddings.shape}")

    # 4. Representation models: POS + KeyBERT
    print("🧠 Building topic representation...")
    pos_model = ensure_pos_model()
    keybert_model = KeyBERTInspired()
    representation_model = {
        "POS": pos_model,
        "KeyBERT": keybert_model
    }

    # 5. Clustering and visualization
    if args.skip_existing and os.path.exists(model_dir):
        print("⚠️ Model already exists. Skipping BERTopic.")
    else:
        print("🚀 Running BERTopic...")
        t2 = time.time()
        st_model = SentenceTransformer(args.embed_model)
        topic_model = BERTopic(
            verbose=True,
            embedding_model=st_model,
            representation_model=representation_model,
            min_topic_size=args.min_topic_size,
        )
        topics, _ = topic_model.fit_transform(convs, embeddings)
        print(f"✅ Clustering completed in {time.time() - t2:.2f} seconds")
        print(f"📚 {len(topic_model.get_topic_info())} topics identified")

        print("🧹 Reducing outliers...")
        new_topics = topic_model.reduce_outliers(convs, topics)

        # Maps topics to conversation_ids
        topic_assignment = {meta[i]["conversation_id"]: new_topics[i] for i in range(len(meta))}
        with open(topic_json_path, "w") as f:
            json.dump(topic_assignment, f, default=str, indent=2)

        print("📊 Generating interactive visualization...")
        fig = topic_model.visualize_topics()
        pio.write_html(fig, file=html_path, auto_open=False)

        print("📁 Saving model and topics...")
        topic_model.save(model_dir, serialization="pytorch", save_ctfidf=True)

        df = topic_model.get_topic_info()
        df.to_csv(topics_csv, index=False)

    print(f"✅ Pipeline completed! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="topic_model_st")
    parser.add_argument("--dataset-name", type=str, default=None, help="Short name for output files")

    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--min-len", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=10000)

    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--min-topic-size", type=int, default=32)
    parser.add_argument("--skip-existing", action="store_true", help="Avoid rework if files already exist")

    args = parser.parse_args()
    main(args)

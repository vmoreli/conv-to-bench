# src/clustering/embeddings.py
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def generate_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 512) -> np.ndarray:
    """
    Generates dense vector embeddings using SentenceTransformers.
    """
    print(f"📅 Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    embeddings = []
    # Use 'desc' in English for the progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="🔐 Generating embeddings"):
        batch = texts[i:i + batch_size]
        # show_progress_bar=False to avoid nested bars
        batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)
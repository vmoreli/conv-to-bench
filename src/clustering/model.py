# src/clustering/model.py
import spacy.cli
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech
from sentence_transformers import SentenceTransformer

def ensure_pos_model(model_name: str = "en_core_web_sm"):
    """Ensures the Spacy Part-of-Speech model is downloaded."""
    try:
        return PartOfSpeech(model_name)
    except OSError:
        print(f"📥 Downloading Spacy model: {model_name}...")
        spacy.cli.download(model_name)
        return PartOfSpeech(model_name)

def train_bertopic(
    documents: list[str], 
    embeddings, 
    embed_model_name: str, 
    min_topic_size: int = 32
):
    """Configures and trains the BERTopic model."""
    
    # 1. Representation Models
    pos_model = ensure_pos_model()
    keybert_model = KeyBERTInspired()
    
    representation_model = {
        "POS": pos_model,
        "KeyBERT": keybert_model
    }

    # 2. Embedding Model (reloaded for internal BERTopic usage)
    st_model = SentenceTransformer(embed_model_name)
    
    # 3. Initialize and Train
    topic_model = BERTopic(
        verbose=True,
        embedding_model=st_model,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
    )
    
    topics, probabilities = topic_model.fit_transform(documents, embeddings)
    return topic_model, topics, probabilities
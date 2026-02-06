# src/data/extraction.py
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import json

def extract_first_user_message(conversation: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extracts the first message sent by the user in a conversation list.
    Assumes the conversation is a list of dicts with "role" and "content" keys.
    """
    if not conversation:
        return None
    
    # Try to get the first message if it is from the user
    first_msg = conversation[0]
    if first_msg.get("role") != "user":
        # If not, search for the first message where role == "user"
        for msg in conversation:
            if msg.get("role") == "user":
                first_msg = msg
                break
        else:
            return None

    content = first_msg.get("content")
    
    # Handle cases where content might be a list (e.g., multimodal) or string
    if isinstance(content, list) and content:
        return str(content[0])
    elif isinstance(content, str):
        return content
    else:
        return str(content)

def extract_conversations_from_dataset(
    dataset, 
    first_n: Optional[int] = None, 
    min_length: int = 32, 
    max_length: int = 10_000
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Iterates through the dataset and extracts the first relevant message.
    Performs simple preprocessing and stores texts and metadata.
    """
    conversations = []
    metadata = []

    # Logic to determine dataset structure (conversation vs message per line)
    is_conversation_per_line = False
    is_message_per_line = False
    
    # [Check for OpenAI Moderation - Logic kept, comments translated]
    if 'openai_moderation' in dataset.column_names:
        print(f"📂 Filtering based on OpenAI Moderation flags.")
        dataset = dataset.filter(lambda x: all(not m["flagged"] for m in x["openai_moderation"]))
    
    # [Structure detection logic]
    if len(dataset) > 0:
        sample_row = dataset[0]
        if "conversation" in sample_row and isinstance(sample_row["conversation"], list):
            is_conversation_per_line = True
        elif "parent_id" in sample_row:
            is_message_per_line = True
        else:
            print("⚠️ Warning: Could not automatically determine dataset format.")
            is_conversation_per_line = True # Default fallback

    # Select subset if first_n is provided
    dataset_iterator = dataset if first_n is None else dataset.select(range(min(first_n, len(dataset))))
    
    if is_message_per_line:
        print("⚙️ Format detected: Message per line. Filtering for root messages (null parent_id)...")
        dataset_iterator = dataset_iterator.filter(lambda x: x.get("parent_id") is None)

    for row_idx, row in enumerate(tqdm(dataset_iterator, desc="🔍 Extracting initial messages")):
        conv_content = None
        conv_id = None

        if is_conversation_per_line:
            # Try different ID fields common in datasets
            conv_id = row.get("conversation_id") or row.get("conversation_hash") or f"conv_idx_{row_idx}"
            conv_content = extract_first_user_message(row["conversation"])
        
        elif is_message_per_line:
            conv_id = row.get("message_tree_id")
            # Handle content extraction for message-per-line
            raw_content = row.get("text") or row.get("content")
            if isinstance(raw_content, list) and raw_content:
                conv_content = str(raw_content[0])
            else:
                conv_content = str(raw_content)

        if not conv_content:
            continue

        # Basic cleanup
        conv_content = conv_content.replace("<|endoftext|>", "<| endoftext |>")
        
        if len(conv_content) <= min_length:
            continue
            
        # Truncate to max length
        conv_content = conv_content[:max_length]

        conversations.append(conv_content)
        metadata.append({"conversation_id": conv_id, "processed_text": conv_content})

    return conversations, metadata


def first_user_message(raw_conversation):
    # Ensure it's in list of dicts format
    if isinstance(raw_conversation, str):
        raw_conversation = json.loads(raw_conversation)
    
    for msg in raw_conversation:
        if msg["role"] == "user":
            return msg["content"].strip()
    return None
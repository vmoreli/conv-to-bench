import os
import json
from typing import List, Optional, Tuple
from datasets import load_from_disk, Dataset

from config import (
    LMSYS_CODING_CONVS_PATH, 
    WILDCHAT_CODING_CONVS_PATH, 
    LMSYS_DS_PATH, 
    WILDCHAT_DS_PATH, 
    LMSYS_FILTERED_PATH, 
    WILDCHAT_FILTERED_PATH
)

def load_datasets(
    lmsys_debug_ids: Optional[List[str]] = None, 
    wildchat_debug_ids: Optional[List[str]] = None
) -> Tuple[Dataset, Dataset]:
    """
    Loads and filters the LMSYS and WildChat datasets.

    If filtered versions already exist on disk, they are loaded directly to save time.
    Otherwise, it loads the raw datasets, filters for coding conversations, and caches the result.

    Args:
        lmsys_debug_ids (list, optional): A list of 'conversation_id' to filter the LMSYS dataset
                                          for debugging purposes. Defaults to None.
        wildchat_debug_ids (list, optional): A list of 'conversation_hash' to filter the WildChat dataset
                                             for debugging purposes. Defaults to None.

    Returns:
        tuple: A tuple containing the two datasets (lmsys_ds, wildchat_ds).
    """
    
    # Check if filtered datasets already exist to load them directly
    if os.path.exists(LMSYS_FILTERED_PATH) and os.path.exists(WILDCHAT_FILTERED_PATH):
        print("📂 Loading pre-filtered datasets from disk...")
        lmsys_ds = load_from_disk(LMSYS_FILTERED_PATH)
        wildchat_ds = load_from_disk(WILDCHAT_FILTERED_PATH)
    else:
        # Load original raw datasets
        print(f"📂 Loading raw LMSYS from: {LMSYS_DS_PATH}")
        lmsys_ds = load_from_disk(LMSYS_DS_PATH)

        print(f"📂 Loading raw WildChat from: {WILDCHAT_DS_PATH}")
        wildchat_ds = load_from_disk(WILDCHAT_DS_PATH)

        # Load coding conversation keys (IDs)
        print(f"📂 Loading LMSYS coding conversation keys from: {LMSYS_CODING_CONVS_PATH}")
        with open(LMSYS_CODING_CONVS_PATH, "r") as f:
            lmsys_coding_conv_keys = set(json.load(f))

        print(f"📂 Loading WildChat coding conversation keys from: {WILDCHAT_CODING_CONVS_PATH}")
        with open(WILDCHAT_CODING_CONVS_PATH, "r") as f:
            wildchat_coding_conv_keys = set(json.load(f))

        # Filter datasets to keep only coding conversations
        print("⚙️  Filtering datasets for coding content...")
        lmsys_ds = lmsys_ds.filter(lambda x: x['conversation_id'] in lmsys_coding_conv_keys)
        wildchat_ds = wildchat_ds.filter(lambda x: x['conversation_hash'] in wildchat_coding_conv_keys)

        # Save filtered datasets for future use
        print("💾 Saving filtered datasets to disk...")
        lmsys_ds.save_to_disk(LMSYS_FILTERED_PATH)
        wildchat_ds.save_to_disk(WILDCHAT_FILTERED_PATH)

    # Apply strict debug filters if specific IDs were provided
    if lmsys_debug_ids:
        print(f"🐛 Running in debug mode for LMSYS with {len(lmsys_debug_ids)} IDs...")
        lmsys_debug_set = set(lmsys_debug_ids)
        lmsys_ds = lmsys_ds.filter(lambda x: x['conversation_id'] in lmsys_debug_set)

    if wildchat_debug_ids:
        print(f"🐛 Running in debug mode for WildChat with {len(wildchat_debug_ids)} IDs...")
        wildchat_debug_set = set(wildchat_debug_ids)
        wildchat_ds = wildchat_ds.filter(lambda x: x['conversation_hash'] in wildchat_debug_set)

    # These limit the dataset to a single entry for quick testing purposes.
    # lmsys_ds = lmsys_ds.select(range(1))
    # wildchat_ds = wildchat_ds.select(range(0)) 

    return lmsys_ds, wildchat_ds
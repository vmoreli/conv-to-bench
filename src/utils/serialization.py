import json
import math

def sanitize_for_json(obj, _path="root"):
    """
    Recursively converts objects to JSON-serializable formats, handling NaNs, 
    bytes, and Pydantic models.
    """
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v, f"{_path}.{k}") for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, "model_dump"):
        return sanitize_for_json(obj.model_dump(), f"{_path}.__model_dump__")
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    
    return str(obj)
def aggregate_token_usage(global_usage: dict, local_usage: dict):
    """Updates global token counters with local usage stats."""
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        global_usage[key] = global_usage.get(key, 0) + local_usage.get(key, 0)

    global_usage["total_cost_usd"] = (
        global_usage.get("total_cost_usd", 0.0) + local_usage.get("total_cost_usd", 0.0)
    )
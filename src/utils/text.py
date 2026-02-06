def format_conversation(raw_conversation):
    # First, convert the string into a list of dictionaries
    if isinstance(raw_conversation, str):
        try:
            raw_conversation = json.loads(raw_conversation)
        except json.JSONDecodeError:
            # Handle cases where the string is not valid JSON
            raise RuntimeError

    formatted_conv = ""
    for i, msg in enumerate(raw_conversation, start=1):
        role = msg['role'].capitalize()
        content = msg['content'].replace('\\n', '\n')
        formatted_conv += f"{i}. **{role}**:\n{content}\n\n"

    return formatted_conv
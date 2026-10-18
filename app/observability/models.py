"""Normalize known usage without inventing missing provider metrics."""


def normalize_usage(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict) or not value:
        return {}
    result = {}
    for target, aliases in {"input": ("input_tokens", "prompt_tokens"), "output": ("output_tokens", "completion_tokens"), "total": ("total_tokens",)}.items():
        for key in aliases:
            if value.get(key) is not None:
                try:
                    count = int(value[key])
                    if count >= 0:
                        result[target] = count
                except (TypeError, ValueError, OverflowError):
                    pass
                break
    return result


def response_usage(response):
    """Accept both LangChain LLMResult and OpenAI ChatCompletion."""
    usage = normalize_usage(getattr(response, "usage", None))
    finish, text = None, None
    choices = getattr(response, "choices", None)
    if choices:
        finish = getattr(choices[0], "finish_reason", None)
        text = getattr(getattr(choices[0], "message", None), "content", None)
    generations = getattr(response, "generations", None)
    if generations and generations[0]:
        generation = generations[0][0]
        message = getattr(generation, "message", None)
        usage = normalize_usage(getattr(message, "usage_metadata", None))
        meta = getattr(message, "response_metadata", None) or {}
        usage = usage or normalize_usage(meta.get("token_usage") or meta.get("usage"))
        text = getattr(message, "content", None)
        finish = meta.get("finish_reason") or (getattr(generation, "generation_info", None) or {}).get("finish_reason")
    output = getattr(response, "llm_output", None) or {}
    usage = usage or normalize_usage(output.get("token_usage") or output.get("usage"))
    return usage, finish, text

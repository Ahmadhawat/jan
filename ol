def ollama_generate(
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 400,
    context: list[int] | None = None,
    timeout: int = 90,
):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if context is not None:
        payload["context"] = context

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    parts = []
    next_context = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue

        obj = json.loads(line)

        if "response" in obj:
            parts.append(obj["response"])

        if obj.get("done") is True:
            next_context = obj.get("context")
            break

    text = "".join(parts).strip()

    if not text:
        raise RuntimeError("Empty response from Ollama")

    return text, next_context
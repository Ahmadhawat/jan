import json
import requests

def ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 400,
    num_ctx: int = 8192,
    timeout: int = 90,
) -> str:
    """
    Call Ollama /api/chat and return the assistant answer as a string.
    """

    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    parts: list[str] = []

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue

        chunk = json.loads(line)

        msg = chunk.get("message")
        if msg and msg.get("content"):
            parts.append(msg["content"])

        if chunk.get("done") is True:
            break

    answer = "".join(parts).strip()

    if not answer:
        raise RuntimeError("Empty response from Ollama")

    return answer
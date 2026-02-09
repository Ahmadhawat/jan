def _post_chat(
    api_url: str,
    model: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 400,
    time_out: int = 90,
) -> str:

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content.strip()},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=time_out)

        if resp.status_code != 200:
            raise RuntimeError(f"[OLLAMA ERROR] {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()

        if not content:
            raise RuntimeError("[LLM ERROR] Leere Antwort vom Ollama-Modell.")

        return content

    except requests.exceptions.ConnectionError:
        raise RuntimeError("[CONNECTION ERROR] Ollama nicht erreichbar (läuft der Server?).")

    except requests.exceptions.ReadTimeout:
        raise RuntimeError("[TIMEOUT] Ollama hat nicht rechtzeitig geantwortet.")

    except Exception as e:
        raise RuntimeError(f"[EXCEPTION] Unerwarteter Fehler: {e}") from e
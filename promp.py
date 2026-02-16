import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """
You are a question answering system.

You must ONLY use the provided documents to answer.

Rules:
- Always cite sources using the provided links
- Do not use outside knowledge
- Do not guess
- If the answer is not in the documents, say exactly:
  "I don't find the answer in the provided documents."
"""

def build_prompt(question, documents):
    context = ""

    for i, doc in enumerate(documents, 1):
        context += f"""
[DOCUMENT {i}]
SOURCE: {doc['url']}
CONTENT:
{doc['text']}
"""

    prompt = f"""
DOCUMENTS:
{context}

---

QUESTION:
{question}

---

INSTRUCTIONS:
- Answer ONLY using the documents above
- Always include source links
- If not found, say: "I don't find the answer in the provided documents."

---

ANSWER:
"""
    return prompt


def query_ollama(question, documents):
    prompt = build_prompt(question, documents)

    payload = {
        "model": "llama3",
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]


# ---- RUN ----
question = "How do you generate text with Ollama?"

answer = query_ollama(question, documents)

print(answer)
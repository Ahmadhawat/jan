import os
import re
import json
import requests

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

DATA_FOLDER = "data"  # folder containing manifest.json
MANIFEST_FILE = os.path.join(DATA_FOLDER, "manifest.json")

TOP_K = 5  # number of chunks to send


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are a question answering system.

You must ONLY use the provided documents.

Each document contains:
- SOURCE: a link
- CONTENT: text

Rules:
- Answer ONLY using the documents
- Always cite sources using (SOURCE: link)
- Do not make up information
- Do not guess
- If the answer is not in the documents, say exactly:
  "I don't find the answer in the provided documents."
"""


# -----------------------------
# PARSE DOCUMENT
# -----------------------------
def parse_document(raw_text):
    # extract source link
    match = re.search(r'<doc ref="([^"]+)">', raw_text)
    source = match.group(1) if match else "UNKNOWN"

    # remove XML tag
    content = re.sub(r'<doc ref="[^"]+">', '', raw_text).strip()

    return {
        "source": source,
        "content": content
    }


# -----------------------------
# LOAD DOCUMENTS FROM MANIFEST
# -----------------------------
def load_documents_from_manifest():
    documents = []

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # iterate in order (Python 3.7+ preserves JSON order)
    for key, value in manifest.items():
        src_path = value.get("src_copy")

        if not src_path or not os.path.exists(src_path):
            print(f"WARNING: File not found -> {src_path}")
            continue

        try:
            with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception as e:
            print(f"ERROR reading {src_path}: {e}")
            continue

        parsed = parse_document(raw_text)

        documents.append(parsed)

    return documents


# -----------------------------
# SIMPLE RETRIEVAL (TEMP)
# -----------------------------
def retrieve_documents(question, documents, top_k=TOP_K):
    """
    Replace later with embeddings
    """
    return documents[:top_k]


# -----------------------------
# BUILD PROMPT
# -----------------------------
def build_prompt(question, documents):
    context = ""

    for i, doc in enumerate(documents, 1):
        context += f"""
[DOCUMENT {i}]
SOURCE: {doc['source']}
CONTENT:
{doc['content']}

"""

    prompt = f"""
DOCUMENTS:
{context}

---

QUESTION:
{question}

---

INSTRUCTIONS:
- Answer ONLY using the documents
- Always cite sources as (SOURCE: link)
- If not found say: "I don't find the answer in the provided documents."

---

ANSWER:
"""
    return prompt


# -----------------------------
# CALL OLLAMA
# -----------------------------
def query_ollama(prompt):
    payload = {
        "model": MODEL,
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

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()["response"]


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def ask(question):
    print(f"\nQuestion: {question}")

    # 1. load docs in correct order
    documents = load_documents_from_manifest()

    print(f"Loaded {len(documents)} documents")

    # 2. retrieve top docs
    top_docs = retrieve_documents(question, documents)

    # 3. build prompt
    prompt = build_prompt(question, top_docs)

    # DEBUG (optional)
    # print(prompt)

    # 4. ask LLM
    answer = query_ollama(prompt)

    print("\nAnswer:")
    print(answer)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    ask("Wie importiere ich Datenart 001?")
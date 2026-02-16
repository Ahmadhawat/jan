import os
import re
import requests

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# change this to your folder
DATA_FOLDER = "data"  

# how many docs to send to LLM
TOP_K = 5  


# -----------------------------
# SYSTEM PROMPT (RULES)
# -----------------------------
SYSTEM_PROMPT = """
You are a question answering system.

You must ONLY use the provided documents to answer.

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
# PARSE FILE
# -----------------------------
def parse_document(raw_text):
    """
    Extract source and content from:
    <doc ref="...">
    """
    # extract source
    match = re.search(r'<doc ref="([^"]+)">', raw_text)
    source = match.group(1) if match else "UNKNOWN"

    # remove xml tag
    content = re.sub(r'<doc ref="[^"]+">', '', raw_text).strip()

    return {
        "source": source,
        "content": content
    }


# -----------------------------
# LOAD FILES
# -----------------------------
def load_documents(folder):
    docs = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        parsed = parse_document(raw_text)
        docs.append(parsed)

    return docs


# -----------------------------
# SIMPLE RETRIEVAL (PLACEHOLDER)
# -----------------------------
def retrieve_documents(question, documents, top_k=TOP_K):
    """
    TODO: replace with vector search
    For now: just return first N docs
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
        raise Exception(f"Ollama error: {response.text}")

    return response.json()["response"]


# -----------------------------
# MAIN RAG FUNCTION
# -----------------------------
def ask(question):
    print(f"\nQuestion: {question}")

    # 1. load documents
    documents = load_documents(DATA_FOLDER)

    # 2. retrieve top docs
    top_docs = retrieve_documents(question, documents)

    # 3. build prompt
    prompt = build_prompt(question, top_docs)

    # DEBUG: print prompt (optional)
    # print(prompt)

    # 4. query model
    answer = query_ollama(prompt)

    print("\nAnswer:")
    print(answer)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    ask("Wie importiere ich Datenart 001?")
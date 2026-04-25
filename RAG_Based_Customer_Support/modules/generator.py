import ollama

def get_llm():
    return "tinyllama"

def generate_answer(llm, query, docs):
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a document extraction assistant.

Extract the COMPLETE answer from the context.

RULES:
- Do NOT summarize
- Do NOT add new information
- Do NOT mix unrelated text
- Return ONLY the exact relevant section
- Preserve numbering and formatting

Context:
{context}

Question:
{query}

Exact Answer:
"""

    response = ollama.chat(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"].strip()

    return answer
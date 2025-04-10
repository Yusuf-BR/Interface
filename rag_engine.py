# ✅ rag_engine.py — handles embedding and retrieval

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path
import requests

# === Paths ===
OUTPUT_FOLDER = Path("outputs")
QNA_PATH = OUTPUT_FOLDER / "generated_qa.json"

# === Load model once ===
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Load Q&A Data ===
def load_qa_data():
    with open(QNA_PATH, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)
    return qa_data

# === Embed Q&A questions ===
def embed_qa_data(qa_data):
    questions = [item['instruction'] for item in qa_data]
    embeddings = model.encode(questions)
    return questions, embeddings

# === Retrieve full Q&A context (Question + Answer + Score) ===
def retrieve_best_context(user_question, questions, embeddings, qa_data, threshold=0.4):
    user_embedding = model.encode([user_question])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < threshold:
        return None, None, None, best_score

    best_qa = qa_data[best_idx]
    best_question = best_qa['instruction']
    best_answer = best_qa['output']

    # Return full context properly
    return best_question, best_answer, best_qa, best_score

# === Call Local LLM via API ===
def call_llm(context, user_question):
    url = "http://localhost:1234/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "meta-llama-3.1-8b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert B2B sales consultant. "
                    "Strictly use the provided context to answer the user's question. "
                    "Do not include the retrieved Q&A pair in your response. "
                    "After answering the question, include professional strategic recommendations "
                    "based on the context, focusing on actionable insights, client prioritization, and outreach strategies. "
                    "If the context does not have the answer, respond clearly: 'I do not have information regarding this in the current knowledge base.' "
                    "Be direct, professional, and consultative."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{user_question}\n\n"
                    "Based strictly on the context, answer the question clearly. "
                    "Then provide strategic recommendations derived from the context."
                )
            }
        ],
        "temperature": 0.0,
        "max_tokens": 800
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    return result['choices'][0]['message']['content'].strip()


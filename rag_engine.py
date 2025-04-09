# ✅ rag_engine.py — handles embedding and retrieval

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path

OUTPUT_FOLDER = Path("outputs")
QNA_PATH = OUTPUT_FOLDER / "generated_qa.json"

# === Load model once ===
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_qa_data():
    with open(QNA_PATH, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)
    return qa_data

def embed_qa_data(qa_data):
    questions = [item['instruction'] for item in qa_data]
    embeddings = model.encode(questions)
    return questions, embeddings

def retrieve_best_answer(user_question, questions, embeddings, qa_data, threshold=0.4):
    user_embedding = model.encode([user_question])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < threshold:
        return "🤖 I couldn't find a relevant answer. Try rephrasing your question."

    return qa_data[best_idx]['output']

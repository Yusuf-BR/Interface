import streamlit as st
import pandas as pd
from pathlib import Path
import rag_engine  # ✅ Import your RAG engine

OUTPUT_FOLDER = Path("outputs")
DATASET_PATH = OUTPUT_FOLDER / "uploaded_dataset.csv"

def show_assistant():
    st.markdown("""
        <h1 style='text-align: center; color: #4A90E2; margin-bottom: 40px;'>
            🤖 B2B Assistant Chat
        </h1>
    """, unsafe_allow_html=True)

    # Check files
    if not DATASET_PATH.exists() or not Path(rag_engine.QNA_PATH).exists():
        st.warning("⚠️ Please upload your dataset and generate Q&A pairs first.")
        return

    # Load and embed Q&A data
    qa_data = rag_engine.load_qa_data()
    questions, embeddings = rag_engine.embed_qa_data(qa_data)

    # User input
    user_question = st.text_input("💬 Enter your question:")

    if user_question:
        with st.spinner("Thinking... 🤖"):
            answer = rag_engine.retrieve_best_answer(user_question, questions, embeddings, qa_data)

        st.markdown(f"### 🧩 Assistant Answer:\n\n{answer}")

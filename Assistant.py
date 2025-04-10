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
            # ✅ Step 1: Retrieve full context (Q + A + Context + Score)
            matched_question, matched_answer, full_context, similarity_score = rag_engine.retrieve_best_context(
                user_question, questions, embeddings, qa_data
            )

            if matched_question is None:
                final_answer = "🤖 I couldn't find a relevant answer in the knowledge base. Try rephrasing your question."
            else:
                # ✅ Step 2: Call LLM with context only (clean prompt)
                final_answer = rag_engine.call_llm(full_context, user_question)

        # ✅ Step 3: Display assistant answer
        st.markdown(f"### 🧩 Assistant Answer:\n\n{final_answer}")

        # ✅ Step 4: Display similarity score for traceability
        if matched_question is not None:
            st.info(f"Similarity Score: {similarity_score:.2f}")

            # ✅ Step 5: Display retrieved Q&A separately (transparency)
            st.markdown("### 🔎 Retrieved Q&A for Reference:")
            st.markdown(f"**Q:** {matched_question}")
            st.markdown(f"**A:** {matched_answer}")

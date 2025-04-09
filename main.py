import streamlit as st
import pandas as pd
from pathlib import Path
from B2b_Qa_Generator import generate_b2b_qa
import Dashboard  # ✅ Manual import of your Dashboard page
import Assistant  # ✅ Manual import of your Assistant page

# === Streamlit page config ===
st.set_page_config(
    page_title="B2B Lead Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === App title ===
st.title("🚀 B2B Lead Assistant: ML + Q&A Automation App")

# === Define paths ===
OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)
QNA_OUTPUT_JSON = OUTPUT_FOLDER / "generated_qa.json"
UPLOADED_CSV_PATH = OUTPUT_FOLDER / "uploaded_dataset.csv"

# === Sidebar navigation ===
st.sidebar.title("📊 B2B Lead Assistant")
page = st.sidebar.radio("Go to", ["Upload Dataset", "Dashboard", "Assistant"])  # ✅ Sidebar option

# === Upload Dataset Page ===
if page == "Upload Dataset":
    st.header("📂 Upload your ML Output Dataset")

    uploaded_file = st.file_uploader("Upload your processed ML output CSV", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Saving file and generating Q&A pairs..."):
            df = pd.read_csv(uploaded_file)
            df.to_csv(UPLOADED_CSV_PATH, index=False)

            # ✅ Generate Q&A pairs
            generate_b2b_qa(input_csv_path=UPLOADED_CSV_PATH, output_json_path=QNA_OUTPUT_JSON)

        st.success("✅ Dataset uploaded and Q&A generated successfully!")
        st.download_button(
            label="📥 Download Generated Q&A JSON",
            data=open(QNA_OUTPUT_JSON, 'rb').read(),
            file_name='generated_qa.json',
            mime='application/json'
        )

        st.markdown("---")
        st.info("Proceed to the Dashboard or Assistant Chat using the sidebar.")
    else:
        st.warning("⚠️ Please upload a CSV file to continue.")

# ✅ Route to Dashboard
elif page == "Dashboard":
    Dashboard.show_dashboard()

# ✅ Route to Assistant Chat (fixed!)
elif page == "Assistant":
    Assistant.show_assistant()

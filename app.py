import streamlit as st
import tempfile
from qa_engine import process_pdf, process_docx, answer_query

# App title
st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("📄 Document-based Q&A")

# File upload
uploaded_file = st.file_uploader(
    "Upload a PDF or Word Document", 
    type=["pdf", "docx"], 
    accept_multiple_files=False
)

# Handle file processing
if uploaded_file is not None:
    # Save to a temp file
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Process document
    with st.spinner("🔍 Processing document..."):
        try:
            if uploaded_file.name.endswith(".pdf"):
                st.session_state.vector_store = process_pdf(file_path)
            elif uploaded_file.name.endswith(".docx"):
                st.session_state.vector_store = process_docx(file_path)
            st.success("✅ Document processed successfully!")
        except Exception as e:
            st.error(f"❌ Error while processing: {e}")

# Ask questions
if "vector_store" in st.session_state:
    query = st.text_input("💬 Ask a question about the document")

    if query:
        with st.spinner("🤖 Generating answer..."):
            try:
                answer = answer_query(st.session_state.vector_store, query)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")


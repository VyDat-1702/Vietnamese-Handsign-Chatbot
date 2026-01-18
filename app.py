import streamlit as st
import os
from RAG_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 RAG Chatbot")
st.divider()

with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    db_exists = os.path.exists("VectorDB_FAISS")
    st.info(f"Vector DB: {'Sẵn sàng' if db_exists else 'Chưa tạo'}")
    
    if st.button("Khởi động Model", disabled=not db_exists, use_container_width=True):
        with st.spinner("Đang load model..."):
            try:
                st.session_state.rag = RAGPipeline()
                st.session_state.rag.initialize()
                st.success(" Sẵn sàng!")
            except Exception as e:
                st.error(f"Lỗi: {e}")
    
    # Clear chat button
    if st.button("🗑️ Xóa chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Help
    with st.expander("Hướng dẫn"):
        st.markdown("""
        **Bước 1:** Tạo Vector DB
        ```bash
        python create_vectordb.py
        ```
        
        **Bước 2:** Chạy app
        ```bash
        streamlit run app.py
        ```
        
        **Bước 3:** Nhấn "Khởi động Model"
        
        **Bước 4:** Bắt đầu chat!
        """)

st.subheader("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn...", disabled=st.session_state.rag is None):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                response = st.session_state.rag.query(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Lỗi: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("RAG Chatbot POC | Powered by LangChain & Streamlit")
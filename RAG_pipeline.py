from unsloth import FastLanguageModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
import os
import torch
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = "VectorDB_FAISS"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "VyDat/qwen3-4b-chat"

PROMPT_TEMPLATE = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời CHỈ dựa trên ngữ cảnh được cung cấp.
Nếu không tìm thấy thông tin trong ngữ cảnh, hãy trả lời theo những gì mà bạn biết một cách tự nhiên.
<|im_end|>
<|im_start|>user
Ngữ cảnh:
{context}

Câu hỏi:
{question}
<|im_end|>
<|im_start|>assistant
"""


class RAGPipeline:
    def __init__(self):
        self.vector_db = None
        self.llm = None
        self.chain = None
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        
        if self.HF_TOKEN is None:
            raise RuntimeError("HF_TOKEN chưa được load từ .env")
    
    def load_vector_db(self):
        if not os.path.exists(VECTOR_DB_PATH):
            raise RuntimeError(
                f"Vector database chưa tồn tại tại '{VECTOR_DB_PATH}'. "
                f"Vui lòng chạy create_vectordb.py trước."
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vector_db = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )
        
        return self.vector_db
    
    def load_llm(self, model_path: str = LLM_MODEL):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            token=self.HF_TOKEN,
            load_in_4bit=False,
        )
        
        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        
        self.llm = HuggingFacePipeline(
            pipeline=model_pipeline,
            model_kwargs={"temperature": 0.7},
        )
        
        return self.llm
    
    def build_chain(self):
        if self.vector_db is None:
            raise RuntimeError("Vector DB chưa được load. Gọi load_vector_db() trước.")
        
        if self.llm is None:
            raise RuntimeError("LLM chưa được load. Gọi load_llm() trước.")
        
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        self.chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
        )

        return self.chain
    
    def initialize(self):
        self.load_vector_db()
        print("Vector database loaded")
        
        self.load_llm()
        print("Language model loaded")
        
        self.build_chain()
        print("RAG chain ready")
        
        return self.chain
    
    def query(self, question: str) -> str:
        if self.chain is None:
            raise RuntimeError("Chain chưa được khởi tạo. Gọi initialize() trước.")
        
        try:
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"Lỗi khi xử lý câu hỏi: {str(e)}"


if __name__ == "__main__":
    print("Khởi động RAG Pipeline...\n")
    
    try:
        rag = RAGPipeline()
        rag.initialize()
        
        print("\n" + "="*60)
        print("RAG Chatbot - Gõ 'quit' để thoát")
        print("="*60 + "\n")
        
        while True:
            question = input("Bạn: ").strip()
            
            if question.lower() in ["quit", "exit", "thoát"]:
                print("Hẹn gặp lại!")
                break
            
            if not question:
                continue
            
            print("\nĐang suy nghĩ...\n")
            answer = rag.query(question)
            print(f"Trợ lý: {answer}\n")
            print("-" * 60 + "\n")
    
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        exit(1)
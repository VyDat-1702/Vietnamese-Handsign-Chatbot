from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from huggingface_hub import login
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not found in .env")

login(token=HF_TOKEN)
print("Login successful!")


def create_vector_db(pdf_dir_path, vectorDB_path):
    loader = DirectoryLoader(
        pdf_dir_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f'\nPhat hien {len(documents)} tai lieu\n')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.from_documents(chunks, embedding)
    db.save_local(vectorDB_path)

    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vector database from PDFs")
    parser.add_argument('--vdb', type=str, default='VectorDB_FAISS', help='Vector database save path')
    parser.add_argument('--dbp', type=str, default='PDF', help='Pdf data path')
    
    args = parser.parse_args()
    
    try:
        create_vector_db(args.dbp, args.vdb)
        print("\nVector database created successfully!")
    except Exception as e:
        print(f"\n Error: {str(e)}")
        exit(1)
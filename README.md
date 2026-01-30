# Installation Guide

## System Requirements

- Python 3.10+
- CUDA 12.+ (if using GPU)
- RAM: Minimum 8GB (Recommended 16GB+)
- GPU: NVIDIA GPU with 8GB+ VRAM (for training)

## Quick Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
conda create -n venv python=3.12
# Activate
conda activate venv
```

### 2. Install Dependencies

**For CPU:**

```bash
pip install -r requirements.txt
```

**For GPU (CUDA):**

```bash
# Edit requirements.txt: uncomment the torch line with cu121
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Install FAISS (Choose GPU/CPU)

**CPU version (default):**

```bash
pip install faiss-cpu
```

**GPU version (faster):**

```bash
pip install faiss-gpu
```

### 4. Install Unsloth (for training)

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Configuration

1. Create `.env` file in root directory:

```bash
HF_TOKEN=your_huggingface_token_here
```

2. Get Hugging Face token:
   - Visit: https://huggingface.co/settings/tokens
   - Create new token with "read" permission
   - Copy and paste into `.env` file

## Project Structure

```
project/
├── .env                  # HF_TOKEN
├── PDF/                  # PDF files directory (create manually)
├── VectorDB_FAISS/       # Vector database (auto-generated)
├── Create_VectorDB.py    # Database creation script
├── RAG_pipeline.py       # Core logic
├── app_local.py                # Streamlit UI
└── requirements.txt      # Dependencies
```

## Verify Installation

```python
# Check torch + CUDA
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check transformers
import transformers
print(f"Transformers version: {transformers.__version__}")

# Check langchain
import langchain
print(f"LangChain version: {langchain.__version__}")
```

## Common Issues & Solutions

### CUDA not available error

```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### FAISS import error

```bash
# Uninstall and reinstall
pip uninstall faiss-cpu faiss-gpu
pip install faiss-cpu  # or faiss-gpu
```

### Out of Memory (OOM) error

- Reduce `chunk_size` in `Create_VectorDB.py`
- Reduce `max_seq_length` in `RAG_pipeline.py`
- Use `load_in_4bit=True` (already default)

## Getting Started

```bash
# Step 1: Create Vector Database
python Create_VectorDB.py

# Step 2: Run Streamlit app
streamlit run app.py
```

## Notes

- Unsloth only supports Linux (WSL2 for Windows)
- Training requires GPU (recommended: A100, V100, or T4)
- For inference only, can run on CPU (slower)

# 📚 SecureDocChat-Secure Multi-PDF Chatbot with LLM Evaluation

A **production-grade Streamlit application** that enables **secure question answering over multiple PDFs** using **Retrieval-Augmented Generation (RAG)** with **Groq-hosted LLMs**.  
The system includes **prompt-injection defense, harmful-content filtering, bias detection**, and **automatic response evaluation**.

---

## 🚀 Features

### 🔍 Retrieval-Augmented Generation (RAG)
- Upload multiple PDFs  
- Automatic chunking and embedding using Sentence Transformers  
- Fast semantic search using FAISS  
- Context-aware responses grounded strictly in retrieved documents  

### 🔐 Security & Safety
- Prompt-injection detection  
- Jailbreak attempt blocking  
- Harmful request filtering  
- Bias detection on both input and output  
- Dual-LLM architecture (separate models for generation & evaluation)  

### 📊 Automatic Evaluation
Each response is evaluated using **LLM-based metrics**:
- **Faithfulness** – grounded in provided context  
- **Coherence** – clarity and logical flow  
- **Completeness** – coverage of the user query  

Scores are displayed directly in the UI.

### 🎛️ Customizable LLM Settings
- Select LLM model for:
  - Response generation  
  - Safety & evaluation  
- Adjustable **temperature (0–1)** via slider  

### 🖥️ Streamlit UI
- Interactive chat interface  
- Sidebar controls  
- Safe deployment on Streamlit Cloud  

---

## 🧠 Supported LLMs (Groq)
- `openai/gpt-oss-20b` *(default)*  
- `llama-3.3-70b-versatile`  
- `qwen/qwen3-32b`  

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit  
- **LLMs:** Groq API  
- **RAG:** LangChain + FAISS  
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`  
- **PDF Parsing:** PyPDFLoader  
- **Evaluation:** LLM-as-a-Judge  

---

## 📂 Project Structure

```text
SecureDocChat/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Cloud-compatible dependencies
├── README.md             # Project documentation
├── .gitignore
└── .streamlit/
    └── secrets.toml      # Local only (NOT committed)
```

## 🔐 Environment Setup
### 1️⃣ Groq API Key
#### Local Development
Create `.streamlit/secrets.toml:`
```toml
GROQ_API_KEY = "gsk_your_groq_api_key"
```
Add to `.gitignore:`
```base
.streamlit/secrets.toml
```
## 📦 Installation (Local)
```base
git clone https://github.com/RathanVeer/SecureDocChat-Secure-Multi-PDF-Chatbot-with-LLM-Evaluation.git
cd SecureDocChat-Secure-Multi-PDF-Chatbot-with-LLM-Evaluation

pip install -r requirements.txt
streamlit run app.py
```
## 🌐 Live Application

You can try out the live Streamlit application here:  
[Secure Multi-PDF RAG Chatbot](https://securedocchat.streamlit.app/)  

Or click the badge to launch:

[![Launch Streamlit App](https://img.shields.io/badge/Launch-Streamlit-blue?logo=streamlit)](https://securedocchat.streamlit.app/)

## 📄 How It Works

1. Upload one or more PDF documents  
2. Click **Process PDFs**  
3. Documents are:
   - Parsed  
   - Chunked  
   - Embedded  
   - Indexed using **FAISS**

4. Ask a question:
   - Input is safety-checked
   - Relevant chunks are retrieved
   - LLM generates an answer grounded in context
   - Output is validated
   - Answer is evaluated and scored

---

## ⚠️ Limitations

- LLM-based evaluation is probabilistic
- Very large PDFs may increase processing time
- FAISS index is in-memory (not persistent across restarts)

---

## 🔮 Future Improvements

- Conversational Mode
- Persistent vector storage (S3 / GCS)
- Streaming responses
- Token & cost tracking
- User authentication
- Feedback-based RLHF
- Multi-language support

---

## 📬 Contact <br>
Rathan Veer <br>
📧 rathanveer.balla@gmail.com <br>
🔗 https://www.linkedin.com/in/rathan-veer-24feb2002/

---

## ⭐ Acknowledgements

- Groq for ultra-fast inference
- LangChain community
- Streamlit team


import streamlit as st
import tempfile
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="📚 Secure RAG Chatbot", layout="wide")
st.title("📚 Secure Multi-PDF RAG Chatbot with Evaluation")

# ---------------- SECRETS ----------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

TOP_K = 4

# ---------------- LLM SELECTION ----------------
available_llms = [
    "openai/gpt-oss-20b",
    "whisper-large-v3",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b"
]

default_llm = "openai/gpt-oss-20b"

st.sidebar.header("🧠 LLM Model Selection")

st.session_state.llm_model = st.sidebar.selectbox(
    "LLM for response generation",
    options=available_llms,
    index=available_llms.index(default_llm)
)

st.session_state.judge_model = st.sidebar.selectbox(
    "LLM for evaluation / classification",
    options=available_llms,
    index=available_llms.index(default_llm)
)

st.sidebar.header("🎛️ Generation Settings")

st.session_state.temperature = st.sidebar.slider(
    "Temperature",
    0.0, 1.0, 0.2, 0.05
)

# ---------------- GROQ UTILS ----------------
def groq_run(model, prompt, temperature=0):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=1024
    )
    return completion.choices[0].message.content.strip()

# ---------------- SECURITY ----------------
def classify(prompt):
    classifier_prompt = f"""
You are a security classifier.

Labels:
- SAFE
- JAILBREAK_ATTEMPT
- HARMFUL_REQUEST
- BIAS

User message:
"{prompt}"

Respond with ONLY one label.
"""
    return groq_run(st.session_state.judge_model, classifier_prompt)

def validate_output(text):
    label = classify(text)
    if label == "HARMFUL_REQUEST":
        return "⚠️ Output blocked for safety reasons."
    if label == "BIAS":
        return "⚠️ Output blocked due to biased content."
    return text

# ---------------- SIDEBAR PDF UPLOAD ----------------
st.sidebar.header("📂 PDF Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload your PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

process_clicked = st.sidebar.button("🚀 Process PDFs")

# ---------------- PROCESS PDFs ----------------
if process_clicked:
    if not uploaded_files:
        st.sidebar.warning("⚠️ Please upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs..."):
            all_docs = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.chat_history = []

        st.sidebar.success("✅ PDFs indexed successfully!")

# ---------------- INIT STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- RAG ----------------
def rag_answer(query):
    label = classify(query)

    if label == "JAILBREAK_ATTEMPT":
        return "🚫 Jailbreak attempt detected.", None, label
    if label == "HARMFUL_REQUEST":
        return "⚠️ Harmful request blocked.", None, label
    if label == "BIAS":
        return "🚫 Biased request detected.", None, label

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the context below.
If not present, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = groq_run(
        st.session_state.llm_model,
        prompt,
        temperature=st.session_state.temperature
    )

    safe_answer = validate_output(answer)

    return safe_answer, context, "SAFE"

# ---------------- EVALUATION ----------------
def geval_score(query, answer, dimension, context):
    prompt = f"""
Evaluate the RESPONSE.

Dimension: {dimension}
Score from 1 to 10.

USER_QUERY:
{query}

RESPONSE:
{answer}

Context:
{context}

Return ONLY:
{{"score": <int>, "explanation": "<short explanation>"}}
"""
    try:
        return eval(groq_run(st.session_state.judge_model, prompt))
    except:
        return {"score": None, "explanation": "Evaluation failed"}

def evaluate_all_metrics(query, answer, context):
    metrics = ["Faithfulness", "Coherence", "Completeness"]
    return {m: geval_score(query, answer, m, context) for m in metrics}

# ---------------- CHAT ----------------
if "vectorstore" not in st.session_state:
    st.info("📄 Upload and process PDFs to start chatting.")
    st.stop()

user_query = st.chat_input("Ask something about your PDFs...")

if user_query:
    with st.spinner("🔐 Safety check..."):
        answer, context, label = rag_answer(user_query)

    scores = None
    if label == "SAFE" and context:
        with st.spinner("📊 Evaluating response..."):
            scores = evaluate_all_metrics(user_query, answer, context)

    st.session_state.chat_history.append({
        "query": user_query,
        "answer": answer,
        "scores": scores,
        "label": label
    })

# ---------------- DISPLAY ----------------
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(item["query"])

    with st.chat_message("assistant"):
        st.write(item["answer"])

        if item["scores"]:
            with st.expander("📊 Evaluation Metrics"):
                for m, r in item["scores"].items():
                    st.markdown(
                        f"**{m}**: `{r['score']}/10`\n\n_{r['explanation']}_"
                    )

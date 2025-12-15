import streamlit as st
import tempfile
import os
import fitz  # PyMuPDF
from groq import Groq

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# ---------------- SESSION STATE INIT ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ---------------- CONFIG ----------------
st.set_page_config(page_title="📚 Secure Multimodal RAG", layout="wide")
st.title("📚 Secure Multi-PDF RAG (Multimodal + Reranking + Token Tracking)")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

TOP_K = 4


# ---------------- MODEL SELECTION ----------------
available_llms = [
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b"
]

default_llm = "openai/gpt-oss-20b"

st.sidebar.header("🤖 LLM Selection")
st.session_state.llm_model = st.sidebar.selectbox(
    "Answer LLM", available_llms, index=0
)
st.session_state.judge_model = st.sidebar.selectbox(
    "Judge / Evaluator LLM", available_llms, index=0
)

st.sidebar.header("⚙️ Generation Settings")
st.session_state.temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.0, 0.2, 0.05
)


# ---------------- COST ESTIMATION ----------------
def estimate_cost(tokens, model):
    PRICING = {
        "llama-3.3-70b-versatile": {"in": 0.60, "out": 0.80},
        "openai/gpt-oss-20b": {"in": 1.00, "out": 3.00},
        "qwen/qwen3-32b": {"in": 0.50, "out": 0.75},
    }

    price = PRICING.get(model)
    if not price:
        return None

    cost = (
        (tokens["input"] / 1_000_000) * price["in"]
        + (tokens["output"] / 1_000_000) * price["out"]
    )
    return round(cost, 6)


# ---------------- GROQ HELPER ----------------
def groq_run(model, prompt, temperature=0):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=1024
    )

    text = completion.choices[0].message.content.strip()
    usage = completion.usage

    return text, usage


# ---------------- SECURITY ----------------
def classify(prompt):
    classifier_prompt = f"""
Classify the user message as one of:
SAFE | JAILBREAK_ATTEMPT | HARMFUL_REQUEST | BIAS

User message:
"{prompt}"

Return only the label.
"""
    label, _ = groq_run(st.session_state.judge_model, classifier_prompt)
    return label.strip()


def validate_output(text):
    label = classify(text)
    if label in ["HARMFUL_REQUEST", "BIAS"]:
        return "⚠️ Output blocked for safety reasons."
    return text


# ---------------- PyMuPDF EXTRACTION ----------------
def extract_text_images_from_pdf(pdf_path, pdf_name):
    doc = fitz.open(pdf_path)
    documents = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        images = page.get_images(full=True)

        image_refs = [f"Image_xref_{img[0]}" for img in images]

        combined = text
        if image_refs:
            combined += "\n\n[IMAGES]\n" + "\n".join(image_refs)

        if combined.strip():
            documents.append(
                Document(
                    page_content=combined,
                    metadata={
                        "source_pdf": pdf_name,
                        "page_number": page_num,
                        "num_images": len(image_refs),
                        "has_images": len(image_refs) > 0
                    }
                )
            )

    return documents


# ---------------- FILE UPLOAD ----------------
uploaded_files = st.sidebar.file_uploader(
    "📄 Upload PDFs (text + images)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("📥 Process PDFs"):
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        all_docs.extend(
            extract_text_images_from_pdf(pdf_path, uploaded_file.name)
        )
        os.remove(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        top_n=TOP_K
    )

    retriever = ContextualCompressionRetriever(
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K * 2}),
        base_compressor=reranker
    )

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.session_state.chat_history = []

    st.sidebar.success("✅ PDFs indexed with metadata & reranking!")


# ---------------- RAG ----------------
def rag_answer(query):
    label = classify(query)
    if label != "SAFE":
        return f"🚫 Blocked: {label}", None, None, label

    docs = st.session_state.retriever.get_relevant_documents(query)

    context = ""
    for d in docs:
        m = d.metadata
        context += (
            f"[Source: {m['source_pdf']} | Page {m['page_number']} "
            f"| Images: {m['num_images']}]\n"
            f"{d.page_content}\n\n"
        )

    prompt = f"""
Answer using ONLY the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    answer, usage = groq_run(
        st.session_state.llm_model,
        prompt,
        temperature=st.session_state.temperature
    )

    return validate_output(answer), context, usage, "SAFE"


# ---------------- EVALUATION ----------------
def geval_score(query, answer, dimension, context):
    prompt = f"""
Evaluate the answer on {dimension}.
Score from 1 to 10.

Query:
{query}

Answer:
{answer}

Context:
{context}

Return JSON:
{{"score": <int>, "explanation": "<short reason>"}}
"""
    try:
        result, _ = groq_run(st.session_state.judge_model, prompt)
        return eval(result)
    except:
        return {"score": None, "explanation": "Evaluation failed"}


def evaluate_all_metrics(query, answer, context):
    return {
        m: geval_score(query, answer, m, context)
        for m in ["Faithfulness", "Coherence", "Completeness"]
    }


# ---------------- CHAT UI ----------------
user_query = st.chat_input("Ask about your PDFs...")

if user_query:
    if st.session_state.retriever is None:
        st.warning("📄 Please upload and process PDFs first.")
    else:
        with st.spinner("🔐 Safety check..."):
            answer, context, usage, label = rag_answer(user_query)

        if label == "SAFE":
            with st.spinner("📊 Evaluating response..."):
                scores = evaluate_all_metrics(user_query, answer, context)
        else:
            scores = None

        tokens = None
        cost = None
        if usage:
            tokens = {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens
            }
            cost = estimate_cost(tokens, st.session_state.llm_model)

        st.session_state.chat_history.append({
            "query": user_query,
            "answer": answer,
            "scores": scores,
            "tokens": tokens,
            "cost": cost
        })


# ---------------- CHAT HISTORY ----------------
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(item["query"])

    with st.chat_message("assistant"):
        st.write(item["answer"])

        if item.get("tokens"):
            st.caption(
                f"🧮 Tokens → In: {item['tokens']['input']} | "
                f"Out: {item['tokens']['output']} | "
                f"Total: {item['tokens']['total']}"
            )

        if item.get("cost") is not None:
            st.caption(f"💰 Estimated cost: ${item['cost']}")

        if item.get("scores"):
            with st.expander("📊 Evaluation Metrics"):
                for m, r in item["scores"].items():
                    st.markdown(
                        f"**{m}**: `{r['score']}/10`\n\n_{r['explanation']}_"
                    )

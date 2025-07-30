import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import docx2txt
import fitz  # PyMuPDF
import re
from io import BytesIO
import os
from dotenv import load_dotenv

# --- Load .env from project root ---
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("OPENROUTER_API_KEY")

# --- Constants ---
REASONING_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# --- Load Preprocessed Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data/Resume/top_candidates.csv")

# --- Extract score from reasoning ---
def extract_score(text):
    total_score_match = re.search(r'Total Score\s*[:\-]?\s*(\d+(\.\d+)?)\s*/\s*100', text, re.IGNORECASE)
    if total_score_match:
        return float(total_score_match.group(1))
    fallback_matches = re.findall(r'(\d+(\.\d+)?)\s*/\s*100', text)
    if fallback_matches:
        return float(fallback_matches[-1][0])
    return None

# --- Classify recommendation ---
def get_recommendation_label(score):
    if score is None:
        return "❓ Unknown"
    elif score >= 85:
        return "✅ Highly Recommended"
    elif score >= 70:
        return "👍 Recommended"
    elif score >= 50:
        return "⚠️ Borderline – Interview with Caution"
    else:
        return "❌ Not Recommended"

# --- Extract positives and negatives ---
def extract_pros_cons(reasoning):
    pros_match = re.search(r"\*\*Strengths:\*\*\s*-([\s\S]*?)(?=\*\*Concerns:|\Z)", reasoning, re.IGNORECASE)
    cons_match = re.search(r"\*\*Concerns:\*\*\s*-([\s\S]*?)(?=(\n\n|\*\*|$))", reasoning, re.IGNORECASE)

    pros = pros_match.group(1).strip() if pros_match else "Not clearly specified"
    cons = cons_match.group(1).strip() if cons_match else "No major issues listed"

    return pros, cons

# --- Extract text from PDF/DOCX ---
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc]).strip()
    elif uploaded_file.name.endswith((".doc", ".docx")):
        return docx2txt.process(BytesIO(uploaded_file.read())).strip()
    else:
        st.error("Unsupported file format. Upload PDF, DOC, or DOCX.")
        return ""

# --- LLM Reasoning Function ---
def get_llm_reasoning(jd, resume_text):
    prompt = f"""
You are a recruiter shortlisting candidates. Here's the job description:

{jd}

Now review the following resume and give a score out of 100 based on how well the candidate matches, prioritizing the skills listed below.
Skills include classroom delivery, curriculum design, tech-enabled instruction, student-centric Science Teaching with hands-on demonstrations.

{resume_text}

Explain the score clearly, including:
- Strengths (positives)
- Weaknesses (areas for improvement)
- Final score (out of 100)
- Recommendation (Strongly Recommended, Recommended, Consider with Caution, Not Recommended)
"""
    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert hiring assistant helping shortlist resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- Job Description ---
science_teacher_jd = """
Looking for a science teacher with experience in student-centric, hands-on classroom delivery, preferably with curriculum design and use of technology.
"""

# --- Load Embedder ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Streamlit UI ---
st.set_page_config(page_title="ATS Resume Shortlisting App", layout="wide")
st.title("📄 ATS Resume Matching & LLM Reasoning")

# --- Load & Display Top Candidates ---
final_df = load_data().copy()
top_5 = final_df.sort_values(by="match_score", ascending=False).head(5)

st.subheader("Top 5 Resume Candidates")
st.dataframe(top_5[["ID", "match_score", "llm_score"]], use_container_width=True)

# --- Semantic Match Bar Chart ---
st.subheader("📊 Semantic Matching Scores")
top_5["Label"] = top_5["ID"].astype(str)
bar_df = top_5.set_index("Label")[["match_score"]]
st.bar_chart(bar_df)

# --- View Reasoning and Resume Preview ---
st.subheader("🧠 LLM Reasoning & Resume Preview")
selected_id = st.selectbox("Select Candidate ID", final_df["ID"])
selected_row = final_df[final_df["ID"] == selected_id].iloc[0]

st.markdown(f"### 🤖 LLM Reasoning\n\n{selected_row['LLM_reasoning']}")
st.markdown(f"### ✅ LLM Score: `{selected_row['llm_score']}/100` | {get_recommendation_label(selected_row['llm_score'])}")

# --- Upload Resume ---
st.subheader("📤 Upload Your Resume (PDF, DOC, DOCX)")
uploaded_file = st.file_uploader("Upload Resume to Evaluate", type=["pdf", "doc", "docx"])

if uploaded_file:
    with st.spinner("Reading and analyzing resume..."):
        resume_text = extract_text(uploaded_file)
        if resume_text:
            jd_embed = embed_model.encode(science_teacher_jd, convert_to_numpy=True)
            res_embed = embed_model.encode(resume_text, convert_to_numpy=True)
            sem_score = cosine_similarity([jd_embed], [res_embed]).flatten()[0]

            llm_reasoning = get_llm_reasoning(science_teacher_jd, resume_text)
            llm_score = extract_score(llm_reasoning)
            recommendation = get_recommendation_label(llm_score)
            pros, cons = extract_pros_cons(llm_reasoning)

            st.markdown(f"### 🔍 Semantic Similarity Score: `{sem_score:.2f}`")
            if llm_score is not None:
                st.markdown(f"### ✅ LLM Score: `{llm_score}/100` | {recommendation}")
            else:
                st.warning("LLM score not extracted properly.")

            st.markdown("### 🧠 LLM Reasoning")
            st.success(llm_reasoning)
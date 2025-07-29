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

# --- Constants ---
REASONING_MODEL = "deepseek/deepseek-chat-v3-0324:free"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# --- Load Preprocessed Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"data/Resume/top_candidates.csv")
    return df

# --- Extract score from reasoning ---
def extract_score(text):
    match = re.search(r'(\d+(\.\d+)?)/100', text)
    return float(match.group(1)) if match else None

# --- Extract text from PDF/DOCX ---
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text() for page in doc])
    elif uploaded_file.name.endswith((".doc", ".docx")):
        text = docx2txt.process(BytesIO(uploaded_file.read()))
    else:
        st.error("Unsupported file format. Upload PDF, DOC, or DOCX.")
        return ""
    return text.strip()

# --- LLM Reasoning Function ---
def get_llm_reasoning(jd, resume_text):
    prompt = f"""
You are a recruiter shortlisting candidates. Here's the job description:

{jd}

Now review the following resume and give a score out of 100 based on how well the candidate matches, prioritizing the skills listed below.
Skills include classroom delivery, curriculum design, tech-enabled instruction, student-centric Science Teaching with hands-on demonstrations.

{resume_text}

Explain your score briefly.
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

selected_id = st.selectbox("Select Candidate ID:", top_5["ID"])
selected_row = final_df[final_df["ID"] == selected_id].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🤖 LLM Reasoning")
    st.info(selected_row.get("LLM_reasoning", "No reasoning found."))

with col2:
    st.markdown("#### 📄 Resume Preview")
    resume_html = selected_row.get("Resume_HTML", "")
    if pd.notna(resume_html) and resume_html.strip():
        st.components.v1.html(resume_html, height=500, scrolling=True)
    else:
        st.warning("No HTML preview found for this candidate.")

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

            st.markdown(f"### 🔍 Semantic Similarity Score: `{sem_score:.2f}`")
            if llm_score is not None:
                st.markdown(f"### ✅ LLM Score: `{llm_score}/100`")
            else:
                st.warning("LLM score not extracted properly.")

            st.markdown("### 🧠 LLM Reasoning")
            st.success(llm_reasoning)
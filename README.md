# 🧠 ATS Resume Shortlisting App

A smart Applicant Tracking System (ATS) powered by **semantic search** and **LLM-based reasoning** to screen and rank resumes against a job description (JD).

🚀 Live Deployment (optional): [Streamlit App Link](#)

## 📌 Project Overview

This tool allows HR teams or hiring managers to:
- Upload a job description and a batch of resumes (PDF, DOC, DOCX).
- Use **Sentence Transformers** to semantically match resumes with JD.
- Leverage an **LLM (DeepSeek via OpenRouter)** to evaluate candidates contextually.
- Generate **reasoning**, assign a **score**, and visualize the **top 5 candidates** with HTML previews of their resumes.

> ⚠️ No manual keyword scanning. It's all AI-based!

---

## ✨ Features

- ✅ **Resume Upload & Preview**
- ✅ **TF-IDF + Semantic Matching**
- ✅ **LLM-based Reasoning (DeepSeek / GPT-style)**
- ✅ **Custom Score Extraction and Ranking**
- ✅ **Experience Classification**
- ✅ **Streamlit UI Dashboard**

---

## 🧰 Tech Stack

| Category       | Tools Used                                     |
|----------------|------------------------------------------------|
| Language       | Python 3.x                                     |
| Embedding      | `sentence-transformers (MiniLM-L6-v2)`         |
| LLM            | DeepSeek via `OpenRouter` API                  |
| Frontend       | Streamlit                                      |
| Matching       | TF-IDF, Cosine Similarity                      |
| File Handling  | PyMuPDF, python-docx, docx2txt                 |
| Visualization  | Streamlit components, matplotlib (optional)    |

---

## 📂 Folder Structure

```bash
NLP-Resume-Parser/
│
├── ats_resume_app.py          # Main Streamlit App
├── Notebook/                  # Development Notebooks
├── data/
│   └── Resume/
│       └── Resume.csv         # Processed Resume Dataset (Filtered)
├── requirements.txt           # Python dependencies
├── README.md                  # You're reading it!
🚀 How to Run Locally

# 1. Create and activate virtual env
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run ats_resume_app.py
📝 Future Improvements
🔄 Add GitHub Actions for auto-deployment

🧠 Replace DeepSeek with self-hosted LLM for private usage

📊 Add resume analysis charts (education level, skill coverage, etc.)

🔒 Add file encryption for resume uploads

🙋‍♂️ Author
Sandipan Dutta
Deep Learning + NLP Enthusiast
GitHub: @sandipanarnab
LinkedIn: https://www.linkedin.com/in/sandipanarnab/

⚠️ Notes
Resume.csv is included only for demo. For production, use your own data.

Ensure OpenRouter API key is active and secure in environment variables.

⭐ Show Some Love
If you like this project, consider ⭐ starring the repo or sharing it with others!

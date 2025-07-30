# 🧠 AI-Powered ATS Resume Shortlisting App

A smart, AI-first Applicant Tracking System (ATS) that automates resume screening using **semantic similarity** and **LLM-based contextual reasoning**—no manual keyword filters!

🔗 **Live Demo**: [Click here to try it on Streamlit →](https://nlp-resume-parser-7a57dyd9wv9jswobchdppc.streamlit.app/)

---

## 📌 What It Does

This AI-driven tool helps recruiters:

- ⚡ Upload a **job description** and a batch of resumes (PDF/DOCX)
- 🧠 Apply **Sentence Transformers** for semantic match scoring
- 🤖 Use **LLM (via OpenRouter)** to evaluate resumes with rich reasoning and a human-like score
- 📊 Visualize the **Top 5 Candidates** with AI-generated analysis and preview

> No keyword matching. Pure AI. Human-level analysis at scale.

---

## 🔍 Features

✅ Resume Upload & PDF/DOCX Parsing  
✅ Semantic Matching via Sentence Transformers  
✅ LLM-based Resume Evaluation (Mistral via OpenRouter)  
✅ Extracted Scoring + AI Reasoning  
✅ Streamlit Interactive UI  
✅ Upload Your Own Resume for Instant Evaluation

---

## 🧰 Tech Stack

| Category       | Tools Used                                 |
|----------------|--------------------------------------------|
| Language       | Python 3.x                                 |
| Embeddings     | `sentence-transformers` (MiniLM-L6-v2)     |
| LLM            | Mistral-7B via OpenRouter API              |
| Frontend       | Streamlit                                  |
| File Parsing   | PyMuPDF, `docx2txt`, `python-docx`         |
| Reasoning      | OpenAI SDK with OpenRouter endpoint        |
| Deployment     | Streamlit Cloud                            |

---

## 🚀 Run It Locally

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenRouter API key to secrets
echo "OPENROUTER_API_KEY=your_key" > .env

# 4. Run the app
streamlit run ats_resume_app.py
```

---

## 📂 Project Structure

```bash
NLP-Resume-Parser/
├── ats_resume_app.py         # Main Streamlit App
├── data/Resume/top_candidates.csv
├── Notebook/ATS Resume parser.ipynb
├── requirements.txt
├── README.md
└── .env                      # (Optional) for local API key use
```

---

## 📈 Future Roadmap

- 🔐 Self-hosted LLM option (e.g., llama.cpp or Ollama)
- 📉 Detailed analytics on skill coverage, experience, gaps
- 🔄 GitHub Actions CI/CD deployment
- 🗂️ Drag-drop bulk resume parsing
- 🔒 Secure upload with encryption for enterprise use

---

## 👨‍💻 About the Developer

**Sandipan Dutta**  
Machine Learning & NLP Enthusiast | Python Developer  
🔗 [LinkedIn](https://www.linkedin.com/in/sandipanarnab/) • 🐙 [GitHub @sandipanarnab](https://github.com/sandipanarnab)

---

## ⭐ Like This Project?

If this project helped you or impressed you, consider giving it a **⭐ star** and sharing with others in the NLP/ATS community!

---

## ⚠️ Notes

- All resumes used are anonymized or demo data.
- API key is securely managed using `.env` or Streamlit secrets.
- Built for educational & prototyping purposes.
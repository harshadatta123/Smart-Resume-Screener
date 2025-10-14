import os
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from groq import Groq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any
from io import BytesIO

# ---------- Load Environment ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file. Please add it.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Resume Scanner (Groq)", layout="wide")
st.title("Resume Scanner — Powered by Groq ⚡")
st.caption("Upload resumes (PDF / TXT / DOCX) and compare them with a Job Description using Groq LLM.")

# ---------- File Reading Utilities ----------

def extract_text_from_pdf_bytes(file_bytes) -> str:
    """Extract text from PDF."""
    reader = PdfReader(stream=file_bytes)
    text = []
    for page in reader.pages:
        try:
            t = page.extract_text()
        except Exception:
            t = ""
        if t:
            text.append(t)
    return "\n\n".join(text)

def extract_text_from_docx(file_bytes) -> str:
    """Extract text from DOCX."""
    doc = Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file) -> str:
    """Determine file type and extract text accordingly."""
    if file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf_bytes(file.getvalue())
    elif file.name.lower().endswith(".txt"):
        return file.getvalue().decode("utf-8", errors="ignore")
    elif file.name.lower().endswith(".docx"):
        return extract_text_from_docx(file.getvalue())
    else:
        raise ValueError("Unsupported file format. Only PDF, TXT, and DOCX are supported.")

# ---------- Groq LLM Helpers ----------

def groq_chat(prompt: str) -> str:
    """Helper to call Groq LLM."""
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_structured_resume(resume_text: str) -> Dict[str, Any]:
    """Use Groq to extract structured resume data."""
    prompt = f"""
    You are a structured data extractor.
    Parse the following resume text and return valid JSON with these fields:
    - name
    - email
    - phone
    - summary
    - skills (list)
    - experience (list of {{title, company, start, end, bullets}})
    - education (list of {{degree, institution, year}})

    Resume text:
    \"\"\"{resume_text[:6000]}\"\"\"
    """
    content = groq_chat(prompt)

    try:
        first, last = content.find("{"), content.rfind("}")
        json_text = content[first:last+1]
        parsed = json.loads(json_text)
    except Exception:
        st.warning("⚠️ JSON parsing failed. Returning minimal structure.")
        parsed = {
            "name": "",
            "email": "",
            "phone": "",
            "summary": resume_text[:500],
            "skills": [],
            "experience": [],
            "education": [],
        }
    return parsed


def groq_score_candidate(candidate_struct: Dict[str, Any], job_desc: str) -> Dict[str, Any]:
    """Use Groq to score candidate against job description."""
    prompt = f"""
    You are an expert recruiter.

    Based on the candidate JSON and job description, return:
    {{
      "match_percent": <integer 0–100>,
      "justification": "<2–3 concise sentences explaining the reasoning>"
    }}

    Candidate JSON:
    {json.dumps(candidate_struct, indent=2)}

    Job Description:
    \"\"\"{job_desc}\"\"\"
    """
    content = groq_chat(prompt)
    try:
        first, last = content.find("{"), content.rfind("}")
        parsed = json.loads(content[first:last+1])
    except Exception:
        parsed = {"match_percent": 50, "justification": "Could not compute automatically."}
    return parsed


def get_simple_embedding(text: str) -> np.ndarray:
    """Use a simple embedding placeholder (Groq currently doesn’t expose embedding models)."""
    # You can replace this with sentence-transformers or OpenAI embedding if needed.
    return np.random.rand(768)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K shortlisted candidates", 1, 20, 5)
    weight_embedding = st.slider("Weight for embedding similarity", 0.0, 1.0, 0.6)
    st.caption("Final Score = (weight × embedding_similarity) + ((1 - weight) × LLM_score / 100)")

# ---------- Main Input ----------
uploaded_files = st.file_uploader("Upload resumes (PDF / TXT / DOCX)", accept_multiple_files=True, type=["pdf", "txt", "docx"])
job_desc = st.text_area("Paste the Job Description", height=250)

# ---------- Main Logic ----------
if st.button("Analyze Resumes"):
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()
    if not job_desc or len(job_desc.strip()) < 20:
        st.error("Please provide a valid job description.")
        st.stop()

    st.info("🔍 Processing resumes using Groq... please wait.")

    job_emb = get_simple_embedding(job_desc)
    results = []

    for f in uploaded_files:
        try:
            resume_text = extract_text(f)
            parsed = extract_structured_resume(resume_text)

            skills_text = " ".join(parsed.get("skills", []))
            exp_text = " ".join([e.get("title", "") + " " + e.get("company", "") for e in parsed.get("experience", [])])
            candidate_text = " ".join([parsed.get("summary", ""), skills_text, exp_text])

            cand_emb = get_simple_embedding(candidate_text)
            sim = cosine_sim(cand_emb, job_emb)
            sim_norm = (sim + 1) / 2

            llm_score = groq_score_candidate(parsed, job_desc)
            llm_percent = llm_score.get("match_percent", 50)
            justification = llm_score.get("justification", "")

            final_score = weight_embedding * sim_norm + (1 - weight_embedding) * (llm_percent / 100)
            final_percent = int(round(final_score * 100))

            results.append({
                "filename": f.name,
                "parsed": parsed,
                "embedding_similarity": sim_norm,
                "llm_percent": llm_percent,
                "justification": justification,
                "final_percent": final_percent
            })

        except Exception as e:
            st.error(f"Error processing {f.name}: {e}")

    # ---------- Display Results ----------
    results_sorted = sorted(results, key=lambda x: x["final_percent"], reverse=True)

    st.header("📋 Shortlisted Candidates")
    for i, r in enumerate(results_sorted[:top_k], start=1):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label=f"{i}. {r['filename']}", value=f"{r['final_percent']}%")
            st.caption(f"Similarity: {r['embedding_similarity']:.3f} | LLM Score: {r['llm_percent']}%")
        with col2:
            parsed = r["parsed"]
            st.subheader(parsed.get("name") or r["filename"])
            st.write(f"**Email:** {parsed.get('email', 'N/A')}")
            st.write(f"**Phone:** {parsed.get('phone', 'N/A')}")
            if parsed.get("summary"):
                st.write("**Summary:**", parsed.get("summary"))
            if parsed.get("skills"):
                st.write("**Skills:**", ", ".join(parsed.get("skills")[:20]))
            st.write("**Justification:**", r["justification"])
            st.markdown("---")

    st.success("✅ Resume analysis complete!")
    st.balloons()

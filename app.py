# app.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in environment (.env). Please set it before running.")
    st.stop()


genai.configure(api_key=GEMINI_API_KEY)


st.set_page_config(page_title="Resume Scanner (Gemini)", layout="wide")
st.title("Resume Scanner — Powered by Gemini Pro ✨")
st.write(
    "Upload resumes (PDF/Text) and paste a job description. "
    "Gemini will extract structured data, compute match scores, and show shortlisted candidates."
)


def extract_text_from_pdf_bytes(file_bytes) -> str:
    reader = PdfReader(stream=file_bytes)
    text = []
    for pg in reader.pages:
        try:
            page_text = pg.extract_text()
        except Exception:
            page_text = ""
        if page_text:
            text.append(page_text)
    return "\n\n".join(text)

def call_gemini_extract_structured(resume_text: str) -> Dict[str, Any]:
    """
    Use Gemini to parse the resume and return structured JSON data.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
    You are a JSON extractor. Extract the following fields from this resume text:

    Fields: name, email, phone, summary, skills (list), experience (list of {{title, company, start, end, bullets}}),
    education (list of {{degree, institution, year}}).

    If a field is missing, leave it empty. Return VALID JSON only.

    Resume text:
    \"\"\"{resume_text[:6000]}\"\"\"
    """
    response = model.generate_content(prompt)
    content = response.text.strip()

    try:
        first = content.find("{")
        last = content.rfind("}")
        json_text = content[first:last+1]
        parsed = json.loads(json_text)
    except Exception:
        st.warning("⚠️ Gemini JSON parsing failed, returning minimal structure.")
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

def get_gemini_embedding(text: str) -> np.ndarray:
    """
    Use Gemini embedding model (if available) to get semantic vector.
    Fallback: random small vector to avoid crash if unavailable.
    """
    try:
        embed_model = "models/text-embedding-004"
        result = genai.embed_content(model=embed_model, content=text)
        emb = np.array(result["embedding"], dtype=np.float32)
        return emb
    except Exception as e:
        st.warning(f"Embedding failed: {e}. Using fallback vector.")
        return np.random.rand(768)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def gemini_score_candidate_against_job(candidate_struct: Dict[str, Any], job_desc: str) -> Dict[str, Any]:
    """
    Use Gemini to compute a recruiter-style score and justification.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
    You are an expert recruiter.

    Given the candidate JSON and a job description, provide a JSON output with:
    - match_percent (integer 0–100)
    - justification (2–3 sentences explaining why)

    Candidate JSON:
    {json.dumps(candidate_struct, indent=2)}

    Job description:
    \"\"\"{job_desc}\"\"\"

    Respond with valid JSON only.
    """
    response = model.generate_content(prompt)
    content = response.text.strip()
    try:
        first = content.find("{")
        last = content.rfind("}")
        parsed = json.loads(content[first:last+1])
    except Exception:
        parsed = {"match_percent": 50, "justification": "Could not compute justification automatically."}
    return parsed


with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K shortlisted to display", min_value=1, max_value=20, value=5)
    weight_embedding = st.slider("Weight for embedding similarity", 0.0, 1.0, 0.6)
    st.markdown("**Final Score** = weight * embedding_similarity + (1 - weight) * (LLM_score / 100)")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description", height=250)

if st.button("Analyze"):
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()
    if not job_desc or len(job_desc.strip()) < 20:
        st.error("Please provide a valid job description.")
        st.stop()

    st.info("🔍 Analyzing resumes... please wait.")
    job_emb = get_gemini_embedding(job_desc)
    results = []

    for f in uploaded_files:
        try:
            file_bytes = f.getvalue()
            text = extract_text_from_pdf_bytes(file_bytes)

            parsed = call_gemini_extract_structured(text)
            summary_text = parsed.get("summary", "")
            skills_text = " ".join(parsed.get("skills", []))
            exp_text = " ".join([e.get("title","") + " " + e.get("company","") for e in parsed.get("experience", [])])
            candidate_text = " ".join([summary_text, skills_text, exp_text])

            cand_emb = get_gemini_embedding(candidate_text)
            sim = cosine_sim(cand_emb, job_emb)
            sim_norm = (sim + 1) / 2

            llm_score = gemini_score_candidate_against_job(parsed, job_desc)
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

    results_sorted = sorted(results, key=lambda x: x["final_percent"], reverse=True)

    st.header("Shortlisted Candidates")
    for i, r in enumerate(results_sorted[:top_k], start=1):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label=f"{i}. {r['filename']}", value=f"{r['final_percent']}%")
            st.caption(f"Similarity: {r['embedding_similarity']:.3f} | Gemini Score: {r['llm_percent']}%")
        with col2:
            parsed = r["parsed"]
            name = parsed.get("name") or r["filename"]
            st.subheader(name)
            st.write("**Email:**", parsed.get("email", "N/A"))
            st.write("**Phone:**", parsed.get("phone", "N/A"))
            if parsed.get("summary"):
                st.write("**Summary:**", parsed.get("summary"))
            if parsed.get("skills"):
                st.write("**Skills:**", ", ".join(parsed.get("skills")[:20]))
            st.write("**Justification:**", r["justification"])
            st.markdown("---")

    st.success("✅ Analysis Complete!")
    st.balloons()

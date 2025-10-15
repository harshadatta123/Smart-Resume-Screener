import os
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any
from io import BytesIO
import base64

def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_local_css("style.css")


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file. Please add it.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)


st.set_page_config(page_title="Resume Scanner (Gemini)", layout="wide")
st.title("Resume Scanner — Powered by Gemini Pro ")
st.caption("Upload resumes (PDF/TXT/DOCX) and compare them with a Job Description.")


def extract_text_from_pdf_bytes(file_bytes) -> str:
    file_stream = BytesIO(file_bytes)
    reader = PdfReader(file_stream)
    text = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            text.append(t)
    return "\n\n".join(text)

def extract_text_from_docx(file_bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file) -> str:
    if file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf_bytes(file.getvalue())
    elif file.name.lower().endswith(".txt"):
        return file.getvalue().decode("utf-8", errors="ignore")
    elif file.name.lower().endswith(".docx"):
        return extract_text_from_docx(file.getvalue())
    else:
        raise ValueError("Unsupported file format. Only PDF, TXT, and DOCX are supported.")


def call_gemini_extract_structured(resume_text: str) -> Dict[str, Any]:
    model = genai.GenerativeModel("gemini-flash-latest")
    prompt = f"""
    Parse this resume text and return JSON with:
    - name, email, phone, summary
    - skills (list)
    - experience (list of {{title, company, start, end, bullets}})
    - education (list of {{degree, institution, year}})
    Resume:
    \"\"\"{resume_text[:6000]}\"\"\""""
    response = model.generate_content(prompt)
    content = response.text.strip()
    try:
        first, last = content.find("{"), content.rfind("}")
        parsed = json.loads(content[first:last+1])
    except Exception:
        parsed = {
            "name": "", "email": "", "phone": "",
            "summary": resume_text[:500], "skills": [],
            "experience": [], "education": [],
        }
    return parsed

def get_gemini_embedding(text: str) -> np.ndarray:
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        return np.array(result["embedding"], dtype=np.float32)
    except Exception as e:
        st.warning(f"Embedding failed: {e}")
        return np.random.rand(768)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(1, -1), b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def gemini_score_candidate_against_job(candidate_struct: dict, job_desc: str) -> dict:
    model = genai.GenerativeModel("gemini-flash-latest")
    skills = " ".join([str(s or "") for s in candidate_struct.get("skills", [])])
    exp = " ".join([f"{str(e.get('title') or '')} {str(e.get('company') or '')}" 
                    for e in candidate_struct.get("experience", []) if isinstance(e, dict)])
    summary = str(candidate_struct.get("summary") or "")
    candidate_text = " ".join([summary, skills, exp]).strip() or "Candidate resume unavailable."

    prompt = f"""
Compare the candidate resume with the job description and output JSON:
{{"match_percent": <0-100>, "justification": "<2-3 concise sentences>"}}
Resume:
\"\"\"{candidate_text}\"\"\"
Job description:
\"\"\"{job_desc}\"\"\""""
    response = model.generate_content(prompt)
    text = response.text.strip()
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last != -1:
        try:
            parsed = json.loads(text[first:last+1])
            parsed["match_percent"] = int(parsed.get("match_percent", 50))
            parsed["justification"] = str(parsed.get("justification", ""))
            return parsed
        except:
            pass
    return {"match_percent": 50, "justification": "Could not compute automatically."}


with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K shortlisted candidates", 1, 20, 5)
    weight_embedding = st.slider("Weight for embedding similarity", 0.0, 1.0, 0.6)
    st.caption("Final = (w × embedding_similarity) + ((1-w) × LLM_score/100)")


uploaded_files = st.file_uploader("Upload resumes", accept_multiple_files=True, type=["pdf", "txt", "docx"])
job_desc = st.text_area("Paste the Job Description", height=250)


if st.button("Analyze Resumes"):
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()
    if not job_desc.strip():
        st.error("Please provide a valid job description.")
        st.stop()

    st.info("🔍 Processing resumes using Gemini Pro... please wait.")
    job_emb = get_gemini_embedding(job_desc)
    results = []

    for f in uploaded_files:
        try:
            resume_text = extract_text(f)
            parsed = call_gemini_extract_structured(resume_text)
            skills_text = " ".join([str(s or "") for s in parsed.get("skills", [])])
            exp_text = " ".join([
                f"{str(e.get('title') or '')} {str(e.get('company') or '')}"
                for e in parsed.get("experience", []) if isinstance(e, dict)
            ])
            summary_text = str(parsed.get("summary") or "")
            candidate_text = " ".join([summary_text, skills_text, exp_text]).strip()
            cand_emb = get_gemini_embedding(candidate_text)
            sim = cosine_sim(cand_emb, job_emb)
            sim_norm = (sim + 1) / 2

            llm_score = gemini_score_candidate_against_job(parsed, job_desc)
            llm_percent = llm_score["match_percent"]
            justification = llm_score["justification"]

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
    st.header("📋 Shortlisted Candidates")


    SAVE_DIR = "shortlisted_resumes"
    os.makedirs(SAVE_DIR, exist_ok=True)
    auto_saved_count = 0

    for r in results_sorted[:top_k]:
        file_obj = next((f for f in uploaded_files if f.name == r["filename"]), None)
        if not file_obj:
            continue

        save_path = os.path.join(SAVE_DIR, file_obj.name)
        with open(save_path, "wb") as f:
            f.write(file_obj.getvalue())

        meta_path = save_path + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        auto_saved_count += 1

    st.success(f" Automatically saved top {auto_saved_count} shortlisted resumes to '{SAVE_DIR}' folder!")


    for i, r in enumerate(results_sorted[:top_k], 1):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(f"{i}. {r['filename']}", f"{r['final_percent']}%")
            st.caption(f"Sim: {r['embedding_similarity']:.3f} | Gemini: {r['llm_percent']}%")
        with col2:
            p = r["parsed"]
            st.subheader(p.get("name") or r["filename"])
            st.write(f"**Email:** {p.get('email', 'N/A')}")
            st.write(f"**Phone:** {p.get('phone', 'N/A')}")
            if p.get("skills"):
                st.write("**Skills:**", ", ".join(p["skills"][:20]))
            st.write("**Justification:**", r["justification"])
            st.markdown("---")

    st.balloons()


SAVE_DIR = "shortlisted_resumes"
st.header("📂 View Saved Resumes")
if os.path.exists(SAVE_DIR):
    saved_files = [f for f in os.listdir(SAVE_DIR) if not f.endswith(".json")]
    if not saved_files:
        st.info("No resumes saved yet.")
    else:
        for file_name in saved_files:
            file_path = os.path.join(SAVE_DIR, file_name)
            meta_path = file_path + ".json"
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            with st.expander(f"📄 {file_name}"):
                st.write(f"**Match Score:** {meta.get('final_percent', 'N/A')}%")
                st.write(f"**Justification:** {meta.get('justification', '')}")
                parsed = meta.get("parsed", {})
                if parsed:
                    st.write(f"**Name:** {parsed.get('name', 'Unknown')}")
                    st.write(f"**Email:** {parsed.get('email', 'N/A')}")
                    if parsed.get("skills"):
                        st.write("**Skills:**", ", ".join(parsed["skills"][:10]))
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">⬇️ Download Resume</a>'
                    st.markdown(href, unsafe_allow_html=True)
else:
    st.info("Run analysis to generate saved resumes.")

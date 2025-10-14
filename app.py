# app.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List

# Load .env (optional)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY in environment or Streamlit secrets.")
openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="Resume Scanner (LLM)", layout="wide")

st.title("Resume Scanner — LLM-powered candidate matching")
st.write("Upload PDF resumes (multiple) and paste a job description. The app extracts structured fields, computes match scores, and shows shortlisted candidates with justification.")

# ---------- Utilities ----------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
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

def call_llm_extract_structured(resume_text: str) -> Dict[str, Any]:
    """Ask the LLM to parse resume_text and return a JSON with structured fields."""
    system_prompt = (
        "You are a helpful parser that reads a candidate resume and extracts structured JSON. "
        "Return valid JSON ONLY. Fields: name, email, phone, summary, skills (list), experience (list of {title, company, start,end, bullets}), education (list of {degree,institution,year}), other (optional)."
    )
    user_prompt = (
        "Extract the following from the resume text below. If a field is missing, return empty string or empty list.\n\n"
        f"RESUME:\n'''{resume_text[:3000]}'''\n\n"
        "Return JSON with keys: name,email,phone,summary,skills,experience,education,other."
    )
    # Use chat completion
    resp = openai.ChatCompletion.create(
        model="gpt-4o", # you can replace with gpt-3.5-turbo if you don't have GPT-4
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=900,
    )
    content = resp["choices"][0]["message"]["content"].strip()

    # The model should return JSON; attempt to parse
    try:
        # find first { and last } to extract JSON block in case model adds commentary
        first = content.find("{")
        last = content.rfind("}")
        json_text = content[first:last+1] if (first != -1 and last != -1) else content
        parsed = json.loads(json_text)
    except Exception as e:
        # fallback: return minimal structure
        st.warning(f"Failed to parse JSON from LLM. Returning minimal structure. Error: {e}")
        parsed = {
            "name": "",
            "email": "",
            "phone": "",
            "summary": resume_text[:500],
            "skills": [],
            "experience": [],
            "education": [],
            "other": ""
        }
    return parsed

# Embeddings helpers
_EMBEDDING_MODEL = "text-embedding-3-small"  # change if desired
def get_embedding(text: str) -> np.ndarray:
    resp = openai.Embedding.create(model=_EMBEDDING_MODEL, input=text)
    emb = np.array(resp["data"][0]["embedding"], dtype=np.float32)
    return emb

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def llm_score_candidate_against_job(candidate_struct: Dict[str, Any], job_desc: str) -> Dict[str, Any]:
    """
    Ask the LLM to provide a match percentage (0-100) and a short justification
    based on the structured fields and the job description.
    """
    prompt = (
        "You are an expert recruiter. Given the candidate structured data (JSON) and a job description, "
        "provide a concise JSON with two fields: 'match_percent' (integer 0-100) and 'justification' (2-4 sentence summary highlighting strengths/gaps). "
        "Base your reasoning on skills, experience, and education. Don't hallucinate details."
        "\n\n"
        f"Candidate JSON:\n{json.dumps(candidate_struct, indent=2)}\n\n"
        f"Job description:\n'''{job_desc}'''\n\n"
        "Return JSON ONLY."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a recruiter assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    content = resp["choices"][0]["message"]["content"].strip()
    try:
        first = content.find("{")
        last = content.rfind("}")
        json_text = content[first:last+1] if (first != -1 and last != -1) else content
        parsed = json.loads(json_text)
    except Exception as e:
        st.warning(f"LLM scoring parse failed: {e}. Returning default.")
        parsed = {"match_percent": 50, "justification": "Could not compute automated justification."}
    return parsed

# ---------- Streamlit UI ----------
with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K shortlisted to display", min_value=1, max_value=20, value=5)
    weight_embedding = st.slider("Weight: embedding similarity (vs LLM score)", min_value=0.0, max_value=1.0, value=0.6)
    st.markdown("Note: final_score = weight_embedding*embedding_similarity + (1-weight_embedding)*(LLM_score/100)")

uploaded_files = st.file_uploader("Upload resumes (PDF) — you can upload multiple", accept_multiple_files=True, type=["pdf"])
job_desc = st.text_area("Paste the Job Description here", height=250)

if st.button("Analyze resumes"):
    if not uploaded_files:
        st.error("Please upload at least one resume (PDF).")
    elif not job_desc or len(job_desc.strip()) < 20:
        st.error("Please paste a reasonably detailed job description.")
    else:
        st.info("Processing... this may take a short while depending on number of resumes.")
        results = []
        # Precompute job embedding
        try:
            job_emb = get_embedding(job_desc)
        except Exception as e:
            st.error(f"Embedding API failed: {e}")
            st.stop()

        for f in uploaded_files:
            try:
                bytes_io = f.getvalue()
                text = extract_text_from_pdf_bytes(bytes_io)
                parsed = call_llm_extract_structured(text)
                # build candidate text summary for embedding: combine skills + summary + job titles
                skills_text = " ".join(parsed.get("skills") if isinstance(parsed.get("skills"), list) else [])
                exp_text = " ".join([
                    (e.get("title","") + " " + e.get("company","") + " " + " ".join(e.get("bullets",[]))) 
                    for e in (parsed.get("experience") or [])
                ])
                candidate_for_emb = " ".join([parsed.get("summary",""), skills_text, exp_text])
                cand_emb = get_embedding(candidate_for_emb[:3000])  # small slice to keep embedding size manageable
                sim = cosine_sim(cand_emb, job_emb)  # in [-1,1]
                # normalize sim to [0,1] (cosine similarity of embeddings is >=0 in many cases but safe to normalize)
                sim_norm = (sim + 1) / 2.0
                # call LLM to get a recruiter-style match percent & justification
                llm_scoring = llm_score_candidate_against_job(parsed, job_desc)
                llm_percent = int(llm_scoring.get("match_percent", 50))
                llm_just = llm_scoring.get("justification", "")

                final_score = weight_embedding * sim_norm + (1 - weight_embedding) * (llm_percent/100.0)
                final_percent = int(round(final_score * 100))

                results.append({
                    "filename": f.name,
                    "parsed": parsed,
                    "embedding_similarity": sim_norm,
                    "llm_percent": llm_percent,
                    "llm_justification": llm_just,
                    "final_percent": final_percent,
                })
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")

        # sort results
        results_sorted = sorted(results, key=lambda x: x["final_percent"], reverse=True)

        st.header("Shortlisted Candidates")
        for idx, r in enumerate(results_sorted[:top_k], start=1):
            col1, col2 = st.columns([1,2])
            with col1:
                st.metric(label=f"{idx}. {r['filename']}", value=f"{r['final_percent']}%")
                st.write(f"Embedding sim: {r['embedding_similarity']:.3f} — LLM score: {r['llm_percent']}%")
            with col2:
                pr = r["parsed"]
                st.subheader(pr.get("name") or r['filename'])
                # Contact
                contact = []
                if pr.get("email"): contact.append(f"Email: {pr.get('email')}")
                if pr.get("phone"): contact.append(f"Phone: {pr.get('phone')}")
                if contact:
                    st.write(", ".join(contact))
                # Summary
                if pr.get("summary"):
                    st.write("**Summary:**", pr.get("summary"))
                # Skills
                skills = pr.get("skills") or []
                if isinstance(skills, list) and skills:
                    st.write("**Skills:**", ", ".join(skills[:30]))
                # Experience (brief)
                exp_list = pr.get("experience") or []
                if exp_list:
                    st.write("**Experience (top entries):**")
                    for e in exp_list[:3]:
                        title = e.get("title","")
                        comp = e.get("company","")
                        st.write(f"- {title} @ {comp} — {e.get('start','')}-{e.get('end','')}")
                # Education
                edu_list = pr.get("education") or []
                if edu_list:
                    st.write("**Education:**", "; ".join([f'{ed.get("degree","")}, {ed.get("institution","")}' for ed in edu_list[:3]]))
                # LLM justification
                st.write("**Recruiter justification:**", r["llm_justification"])
                st.markdown("---")

        st.success("Analysis complete.")
        st.balloons()

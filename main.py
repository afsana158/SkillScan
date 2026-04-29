from fastapi import FastAPI, File, UploadFile, Form
import fitz
from docx import Document
import requests
import json
import os
import re

app = FastAPI(title="Resume Analyzer")

FREE_MODELS = [
    "google/gemma-3-27b-it:free",
    "google/gemma-4-31b-it:free",
    "mistralai/mistral-small-3.1:free",
    "nvidia/llama-3.1-nemotron-70b-instruct:free",
]

# ── Extract Text ──────────────────────────────────────────
def extract_text(file: UploadFile) -> str:
    if file.filename.endswith(".pdf"):
        pdf = fitz.open(stream=file.file.read(), filetype="pdf")
        return "".join(p.get_text() for p in pdf)

    elif file.filename.endswith(".docx"):
        doc = Document(file.file)
        return "\n".join(p.text for p in doc.paragraphs)

    elif file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8", errors="ignore")

    return ""

# ── Keyword Extractor (Fallback) ──────────────────────────
def extract_keywords(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    words = set(text.split())
    return {w for w in words if len(w) > 2}

# ── ATS Fallback Scorer ───────────────────────────────────
def ats_fallback(resume_text: str, jd_text: str) -> dict:
    resume_words = extract_keywords(resume_text)
    jd_words = extract_keywords(jd_text)

    if not jd_words:
        return {
            "candidate_name": "Unknown",
            "match_score": 0,
            "missing_skills": [],
            "summary": "Job description is empty or invalid."
        }

    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words

    score = int((len(matched) / len(jd_words)) * 100)
    missing_skills = list(missing)[:8]

    return {
        "candidate_name": "Unknown",
        "match_score": score,
        "missing_skills": missing_skills,
        "summary": f"Fallback ATS scoring based on keyword overlap. Matched {len(matched)} out of {len(jd_words)} relevant terms."
    }

# ── JSON Extractor ────────────────────────────────────────
def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        print("EXTRACT_JSON FAILED — raw content:", text[:300])
        raise ValueError("No JSON object found")

    return json.loads(text[start:end])

# ── AI Analyzer ───────────────────────────────────────────
def analyze_with_ai(resume_text: str, jd_text: str) -> dict:
    prompt = f"""You are a strict ATS (Applicant Tracking System).

Analyze the resume against the job description. Return ONLY a JSON object.

Rules:
- match_score: integer 0-100 (no decimals)
- missing_skills: max 8 items, 1-3 words each
- candidate_name: extract from resume header or "Unknown"
- summary: 1-2 sentences

Return JSON:
{{
  "candidate_name": "string",
  "match_score": number,
  "missing_skills": ["skill1"],
  "summary": "string"
}}

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:1500]}"""

    for model in FREE_MODELS:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPEN_ROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=45
            )

            res_json = response.json()
            print(f"MODEL: {model} | STATUS: {response.status_code}")

            if "error" in res_json:
                print(f"Rate limited on {model}: {res_json['error']}")
                continue

            if not res_json.get("choices"):
                print(f"No choices from {model}")
                continue

            content = res_json["choices"][0]["message"]["content"]
            result = extract_json(content)

            score = float(result.get("match_score", 0))
            if 0 < score <= 1:
                score *= 100
            result["match_score"] = int(round(score))

            print(f"Success with model: {model}")
            return result

        except Exception as e:
            print(f"Error with {model}: {e}")
            continue

    # ── FALLBACK TRIGGER ───────────────────────────────
    print("All AI models failed → using ATS fallback")
    return ats_fallback(resume_text, jd_text)

# ── Endpoint ──────────────────────────────────────────────
@app.post("/review")
async def review_resume(
    file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    resume_text = extract_text(file)

    if not resume_text.strip():
        return {
            "candidate_name": "Unknown",
            "match_score": 0,
            "missing_skills": [],
            "summary": "Could not extract text from the uploaded file."
        }

    return analyze_with_ai(resume_text, jd_text)

@app.get("/ping")
def ping():
    return {"status": "alive"}

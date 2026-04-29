from fastapi import FastAPI, File, UploadFile, Form
import fitz
from docx import Document
import requests
import json
import os
import re

app = FastAPI(title="Resume Analyzer")

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

# ── JSON Extractor ────────────────────────────────────────
def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()  # strip markdown fences
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])

# ── AI Call ───────────────────────────────────────────────
def analyze_with_ai(resume_text: str, jd_text: str) -> dict:
    prompt = f"""You are a strict ATS (Applicant Tracking System).

Analyze the resume against the job description. Return ONLY a JSON object, no explanation, no markdown.

Rules:
- match_score: integer 0-100. Never a decimal like 0.85, always like 85.
- missing_skills: specific technical skills from the JD not found in the resume. Max 8 items, 1-3 words each.
- candidate_name: extract from resume header. If not found, use "Unknown".
- summary: 1-2 sentences explaining the score. Be specific.

Return ONLY this JSON:
{{
  "candidate_name": "string",
  "match_score": number,
  "missing_skills": ["skill1", "skill2"],
  "summary": "string"
}}

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:1500]}"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPEN_ROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "temperature": 0,   # ✅ deterministic output
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=45   # ✅ handles Render cold starts
        )

        print("STATUS:", response.status_code)
        print("RAW:", response.text[:500])

        content = response.json()["choices"][0]["message"]["content"]
        result = extract_json(content)

        # Normalize score defensively
        score = float(result.get("match_score", 0))
        if 0 < score <= 1:
            score = score * 100
        result["match_score"] = int(round(score))

        return result

    except Exception as e:
        print("AI Error:", e)
        return {
            "candidate_name": "Unknown",
            "match_score": 0,
            "missing_skills": [],
            "summary": f"Analysis failed: {str(e)}"
        }

# ── Endpoint ──────────────────────────────────────────────
@app.post("/review")
async def review_resume(
    file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    resume_text = extract_text(file)
    return analyze_with_ai(resume_text, jd_text)
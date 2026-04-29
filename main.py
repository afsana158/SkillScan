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

# ── JSON Extractor ────────────────────────────────────────
def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        print("EXTRACT_JSON FAILED — raw content:", text[:300])
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
                print(f"Rate limited on {model}, trying next... Error: {res_json['error']}")
                continue

            if not res_json.get("choices"):
                print(f"No choices from {model}, trying next...")
                continue

            content = res_json["choices"][0]["message"]["content"]
            result = extract_json(content)

            score = float(result.get("match_score", 0))
            if 0 < score <= 1:
                score *= 100
            result["match_score"] = int(round(score))

            print(f"Success with model: {model}, score: {result['match_score']}")
            return result

        except Exception as e:
            print(f"Error with {model}: {e}")
            continue

    return {
        "candidate_name": "Unknown",
        "match_score": 0,
        "missing_skills": [],
        "summary": "All models are currently rate-limited. Please try again in a few minutes."
    }

# ── Endpoint ──────────────────────────────────────────────
@app.post("/review")
async def review_resume(
    file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    resume_text = extract_text(file)
    return analyze_with_ai(resume_text, jd_text)

@app.get("/ping")
def ping():
    return {"status": "alive"}
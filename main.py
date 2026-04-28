from fastapi import FastAPI, File, UploadFile, Form
import fitz  # PyMuPDF
from docx import Document
import requests
import json
import os

app = FastAPI(title="Lightweight Resume Analyzer")

# -------------------------------
# Extract text from files
# -------------------------------
def extract_text(file: UploadFile) -> str:
    if file.filename.endswith(".pdf"):
        text = ""
        pdf = fitz.open(stream=file.file.read(), filetype="pdf")
        for p in pdf:
            text += p.get_text()
        return text

    elif file.filename.endswith(".docx"):
        doc = Document(file.file)
        return "\n".join([p.text for p in doc.paragraphs])

    elif file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8", errors="ignore")

    return ""


# -------------------------------
# AI CALL
# -------------------------------
def analyze_with_ai(resume_text, jd_text):
    prompt = f"""
You are an AI recruiter.

Analyze the resume and job description.

Return ONLY valid JSON:

{{
  "candidate_name": "",
  "match_score": number,
  "missing_skills": [],
  "summary": ""
}}

Resume:
{resume_text[:2000]}

Job Description:
{jd_text}
"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPEN_ROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )

        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]

        # Clean JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        content = content[start:end]

        return json.loads(content)

    except Exception as e:
        print("AI Error:", e)

        return {
            "candidate_name": "Unknown",
            "match_score": 0,
            "missing_skills": [],
            "summary": "Fallback response due to error"
        }


# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/review")
async def review_resume(
    file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    resume_text = extract_text(file)

    result = analyze_with_ai(resume_text, jd_text)

    return result
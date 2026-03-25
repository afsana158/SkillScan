from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import datetime
import re
import json
import spacy
from pydantic import BaseModel
import fitz
from docx import Document
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="Resume Skill Gap Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SECTION_HEADERS = [
    "objective", "summary", "education", "experience", "work experience",
    "projects", "skills", "certifications", "certificates", "achievements"
]

# All keys lowercase for consistent matching
ROLE_SKILLS = {
    "frontend developer": [
        "html", "css", "javascript", "react", "redux", "typescript", "next.js",
        "vue", "angular", "tailwind", "bootstrap", "responsive design", "git", "rest api"
    ],
    "backend developer": [
        "python", "java", "node.js", "express", "django", "flask",
        "sql", "postgresql", "mysql", "mongodb", "aws", "docker",
        "rest api", "linux", "redis", "kubernetes"
    ],
    "ai engineer": [
        "python", "tensorflow", "pytorch", "machine learning", "deep learning",
        "nlp", "opencv", "transformers", "pandas", "numpy", "scikit-learn", "computer vision"
    ],
    "data analyst": [
        "python", "sql", "excel", "pandas", "numpy", "matplotlib",
        "seaborn", "power bi", "tableau", "data visualization",
        "statistics", "reporting", "dashboard"
    ],
    "devops engineer": [
        "docker", "kubernetes", "aws", "ci/cd", "jenkins", "terraform",
        "ansible", "linux", "bash", "monitoring", "prometheus", "grafana"
    ],
    "full stack developer": [
        "html", "css", "javascript", "react", "node.js", "express",
        "mongodb", "sql", "rest api", "git", "docker", "aws"
    ],
}

FUTURE_SKILL_MAP = {
    "python": ["Django", "Flask", "FastAPI", "Pandas", "Numpy"],
    "machine learning": ["Deep learning", "MLOps", "Feature engineering", "Transformers"],
    "deep learning": ["PyTorch", "Tensorboard", "Model optimization"],
    "javascript": ["TypeScript", "Next.js", "React"],
    "sql": ["Data warehousing", "BigQuery", "ETL pipelines"],
    "react": ["Next.js", "Redux", "TypeScript"],
    "docker": ["Kubernetes", "CI/CD", "AWS ECS"],
    "node.js": ["NestJS", "GraphQL", "Microservices"],
    "aws": ["Terraform", "CloudFormation", "Serverless"],
}

LEARNING_RESOURCES = {
    "docker": {"url": "https://docs.docker.com/get-started/", "platform": "Official Docs"},
    "kubernetes": {"url": "https://kubernetes.io/docs/tutorials/", "platform": "Official Docs"},
    "aws": {"url": "https://aws.amazon.com/training/", "platform": "AWS Training"},
    "react": {"url": "https://react.dev/learn", "platform": "Official Docs"},
    "typescript": {"url": "https://www.typescriptlang.org/docs/", "platform": "Official Docs"},
    "postgresql": {"url": "https://www.postgresqltutorial.com/", "platform": "PostgreSQL Tutorial"},
    "redis": {"url": "https://redis.io/learn", "platform": "Redis University"},
    "machine learning": {"url": "https://www.coursera.org/learn/machine-learning", "platform": "Coursera"},
    "pytorch": {"url": "https://pytorch.org/tutorials/", "platform": "Official Docs"},
    "tensorflow": {"url": "https://www.tensorflow.org/tutorials", "platform": "Official Docs"},
    "sql": {"url": "https://mode.com/sql-tutorial/", "platform": "Mode Analytics"},
    "tableau": {"url": "https://www.tableau.com/learn/training", "platform": "Tableau"},
    "mongodb": {"url": "https://learn.mongodb.com/", "platform": "MongoDB University"},
    "next.js": {"url": "https://nextjs.org/learn", "platform": "Official Docs"},
    "tailwind": {"url": "https://tailwindcss.com/docs", "platform": "Official Docs"},
    "django": {"url": "https://docs.djangoproject.com/en/stable/intro/tutorial01/", "platform": "Official Docs"},
    "flask": {"url": "https://flask.palletsprojects.com/en/stable/tutorial/", "platform": "Official Docs"},
    "linux": {"url": "https://linuxjourney.com/", "platform": "Linux Journey"},
    "git": {"url": "https://learngitbranching.js.org/", "platform": "Learn Git Branching"},
    "power bi": {"url": "https://learn.microsoft.com/en-us/power-bi/", "platform": "Microsoft Learn"},
}

SKILL_CATEGORIES = {
    "Programming Languages": ["python", "java", "javascript", "typescript", "c++", "c", "golang", "rust", "kotlin"],
    "Frontend": ["react", "vue", "angular", "next.js", "html", "css", "tailwind", "bootstrap", "redux"],
    "Backend": ["node.js", "express", "django", "flask", "fastapi", "spring boot", "nestjs"],
    "Databases": ["sql", "mysql", "postgresql", "mongodb", "redis", "firebase", "sqlite"],
    "Cloud & DevOps": ["aws", "docker", "kubernetes", "ci/cd", "jenkins", "terraform", "linux", "git"],
    "AI & ML": ["machine learning", "deep learning", "pytorch", "tensorflow", "nlp", "opencv",
                 "scikit-learn", "pandas", "numpy", "transformers", "computer vision"],
    "Data & Analytics": ["tableau", "power bi", "excel", "data visualization", "statistics", "reporting", "dashboard"],
}

# Add this helper — scans raw text for any known skill keywords
ALL_KNOWN_SKILLS = sorted(set(
    skill for skills in ROLE_SKILLS.values() for skill in skills
    ) | set(
    skill for skills in SKILL_CATEGORIES.values() for skill in skills
))

def extract_known_skills_from_text(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for skill in ALL_KNOWN_SKILLS:
        # Use word boundary to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


# ──────────────────────────────────────────────
# File Text Extraction
# ──────────────────────────────────────────────
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
    else:
        raise HTTPException(400, "Unsupported file type. Only PDF, DOCX, TXT allowed.")


# ──────────────────────────────────────────────
# Section Splitter
# ──────────────────────────────────────────────
def simple_section_split(text: str) -> Dict[str, str]:
    text_lower = text.lower()
    header_pattern = r"(^|\n)\s{0,20}(" + "|".join(SECTION_HEADERS) + r")\s*[:\-\n]"
    parts = {}
    matches = list(re.finditer(header_pattern, text_lower, re.IGNORECASE | re.MULTILINE))
    if matches:
        indices = [(m.start(), m.group(2).strip().lower()) for m in matches]
        for i, (start, header) in enumerate(indices):
            next_start = indices[i + 1][0] if i + 1 < len(indices) else len(text)
            section = text[start:next_start].strip()
            section = "\n".join(section.splitlines()[1:])
            parts[header] = section
        return parts
    return {"objective": text}


# ──────────────────────────────────────────────
# Skill Extractor + Categorizer
# ──────────────────────────────────────────────
def categorize_skills(skills: List[str]) -> Dict[str, List[str]]:
    result = {}
    skills_lower = [s.lower() for s in skills]
    for category, keywords in SKILL_CATEGORIES.items():
        matched = [s for s in skills_lower if any(k in s for k in keywords)]
        if matched:
            result[category] = matched
    return result


# ──────────────────────────────────────────────
# Section Parsers
# ──────────────────────────────────────────────
def extract_experience(section: str) -> List[Dict]:
    result = []
    lines = [l.strip() for l in section.splitlines() if l.strip()]
    current = None
    for l in lines:
        if re.search(r"(developer|intern|engineer|analyst|manager|at\s)", l.lower()):
            result.append({"title": l, "description": ""})
            current = result[-1]
        elif current:
            current["description"] += " " + l
    return result


def extract_projects(section: str) -> List[Dict]:
    result = []
    lines = [l.strip() for l in section.splitlines() if l.strip()]
    current = None
    for l in lines:
        if len(l.split()) <= 10:
            result.append({"title": l, "description": ""})
            current = result[-1]
        elif current:
            current["description"] += " " + l
    return result


def extract_education(section: str) -> List[Dict]:
    result = []
    lines = [l.strip() for l in section.splitlines() if l.strip()]
    for l in lines:
        if re.search(r"(b\.tech|bachelor|m\.tech|master|phd|b\.e|b\.sc|mca|bca|20\d{2})", l.lower()):
            ents = [ent.text for ent in nlp(l).ents if ent.label_ in ("ORG", "GPE")]
            result.append({"line": l, "institution": ents[0] if ents else ""})
    return result


def extract_certifications(section: str) -> List[str]:
    return [l.strip() for l in section.splitlines() if l.strip() and len(l.strip()) > 3]


# ──────────────────────────────────────────────
# Scoring Functions
# ──────────────────────────────────────────────
def compute_fit_score(resume_text: str, job_role: str) -> float:
    job_role = job_role.lower().strip()
    job_skills = ROLE_SKILLS.get(job_role, [])
    if not job_skills:
        return 0.0

    # Skill overlap ratio (primary signal)
    resume_skills_lower = extract_known_skills_from_text(resume_text)
    job_skills_lower = {s.lower() for s in job_skills}
    matched = resume_skills_lower & job_skills_lower
    skill_overlap_score = len(matched) / len(job_skills_lower)  # 0.0 – 1.0

    # BERT semantic similarity (secondary signal)
    job_text = " ".join(job_skills)
    resume_emb = bert_model.encode(resume_text, convert_to_tensor=True)
    job_emb = bert_model.encode(job_text, convert_to_tensor=True)
    bert_score = util.pytorch_cos_sim(resume_emb, job_emb).item()  # 0.0 – 1.0

    # Weighted blend: 70% skill overlap, 30% semantic similarity
    blended = (0.7 * skill_overlap_score) + (0.3 * bert_score)
    return round(blended * 100, 2)


def compute_resume_strength(sections: Dict, text: str) -> Dict:
    action_verbs = [
        "built", "developed", "led", "designed", "optimized", "implemented",
        "created", "improved", "managed", "deployed", "architected", "automated",
        "reduced", "increased", "launched", "delivered", "engineered", "integrated"
    ]
    verb_count = sum(1 for v in action_verbs if v in text.lower())
    number_count = len(re.findall(r'\b\d+%?\b', text))
    section_score = sum(1 for s in ["education", "experience", "projects", "skills"] if s in sections)
    all_skills = extract_known_skills_from_text(text)
    skill_density = round(len(all_skills) / max(len(text.split()), 1) * 100, 2)

    return {
        "action_verb_score": min(verb_count * 10, 100),
        "quantification_score": min(number_count * 8, 100),
        "section_completeness": section_score * 25,
        "skill_density": min(skill_density * 10, 100),
        "overall": round(
            (min(verb_count * 10, 100) +
             min(number_count * 8, 100) +
             section_score * 25 +
             min(skill_density * 10, 100)) / 4, 2
        )
    }


def get_feedback(score: float) -> str:
    if score >= 75:
        return "Strong resume match! Focus on quantifying your achievements more."
    elif score >= 50:
        return "Decent match. Add more role-specific skills and project outcomes."
    else:
        return "Weak match for this role. Tailor your resume with relevant skills and projects."


# ──────────────────────────────────────────────
# Recommendation Helpers
# ──────────────────────────────────────────────
def predict_future_skills(current_skills: List[str]) -> List[str]:
    future = set()
    skills_lower = [s.lower() for s in current_skills]
    for skill in skills_lower:
        if skill in FUTURE_SKILL_MAP:
            future.update(FUTURE_SKILL_MAP[skill])
    return sorted(list(future))


def predict_future_roles(skills: List[str], top_n: int = 3) -> List[Dict]:
    skill_text = " ".join([s.lower() for s in skills])
    skill_emb = bert_model.encode(skill_text, convert_to_tensor=True)
    role_scores = []
    for role, role_skills in ROLE_SKILLS.items():
        role_text = " ".join(role_skills)
        role_emb = bert_model.encode(role_text, convert_to_tensor=True)
        score = round(util.pytorch_cos_sim(skill_emb, role_emb).item() * 100, 2)
        role_scores.append({"role": role.title(), "match_score": score})
    role_scores.sort(key=lambda x: x["match_score"], reverse=True)
    return role_scores[:top_n]


def get_resources_for_missing(missing_skills: List[str]) -> List[Dict]:
    result = []
    for skill in missing_skills:
        key = skill.lower()
        if key in LEARNING_RESOURCES:
            result.append({"skill": skill, **LEARNING_RESOURCES[key]})
        else:
            result.append({
                "skill": skill,
                "url": f"https://www.youtube.com/results?search_query=learn+{skill.replace(' ', '+')}",
                "platform": "YouTube"
            })
    return result


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.post("/review")
async def review_resume(
    file: UploadFile = File(...),
    job_role: str = Form("frontend developer")
):
    """Analyze resume against a predefined job role."""
    text = extract_text(file)
    job_role = job_role.lower().strip()
    sections = simple_section_split(text)

    parsed = {
        "education": extract_education(sections.get("education", "")),
        "experience": extract_experience(sections.get("experience", sections.get("work experience", ""))),
        "projects": extract_projects(sections.get("projects", "")),
        "skills": list(extract_known_skills_from_text(text)),
        "certifications": extract_certifications(
            sections.get("certifications", sections.get("certificates", ""))
        ),
    }

    score = compute_fit_score(text, job_role)
    skills_lower = {s.lower() for s in parsed["skills"]}
    expect_lower = {s.lower() for s in ROLE_SKILLS.get(job_role, [])}
    skills_matched = sorted(list(skills_lower & expect_lower))
    skills_missing = sorted(list(expect_lower - skills_lower))

    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "job_role": job_role,
        "fit_score": score,
        "skills_matched": skills_matched,
        "skills_missing": skills_missing,
        "skills_categorized": categorize_skills(parsed["skills"]),
        "resume_strength": compute_resume_strength(sections, text),
        "learning_resources": get_resources_for_missing(skills_missing),
        "future_role_suggestions": predict_future_roles(parsed["skills"]),
        "feedback": get_feedback(score),
        "analysis": parsed,
    }


@app.post("/review-jd")
async def review_with_jd(
    file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    """Analyze resume against a pasted Job Description (real JD, not predefined role)."""
    resume_text = extract_text(file)
    sections = simple_section_split(resume_text)

    # BERT similarity against actual JD text
    jd_skills = extract_known_skills_from_text(jd_text)
    resume_skills = extract_known_skills_from_text(resume_text)

    skill_overlap_score = len(jd_skills & resume_skills) / max(len(jd_skills), 1)

    resume_emb = bert_model.encode(resume_text, convert_to_tensor=True)
    jd_emb = bert_model.encode(jd_text, convert_to_tensor=True)
    bert_score = util.pytorch_cos_sim(resume_emb, jd_emb).item()

    score = round(((0.7 * skill_overlap_score) + (0.3 * bert_score)) * 100, 2)

    # Skill comparison
    jd_skills = extract_known_skills_from_text(jd_text)
    resume_skills = extract_known_skills_from_text(resume_text)
    matched = sorted(list(jd_skills & resume_skills))
    missing = sorted(list(jd_skills - resume_skills))

    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "fit_score": score,
        "jd_skill_count": len(jd_skills),
        "resume_skill_count": len(resume_skills),
        "skills_matched": matched,
        "skills_missing": missing,
        "skills_categorized": categorize_skills(list(resume_skills)),
        "resume_strength": compute_resume_strength(sections, resume_text),
        "learning_resources": get_resources_for_missing(missing[:10]),
        "feedback": get_feedback(score),
    }


@app.post("/compare-jds")
async def compare_multiple_jds(
    file: UploadFile = File(...),
    jd_list: str = Form(...)   # JSON string: [{"title": "ML Eng @ Zepto", "text": "..."}]
):
    """Compare resume against multiple JDs and rank them."""
    try:
        jds = json.loads(jd_list)
    except json.JSONDecodeError:
        raise HTTPException(400, "jd_list must be a valid JSON array.")

    resume_text = extract_text(file)
    resume_emb = bert_model.encode(resume_text, convert_to_tensor=True)

    results = []
    for jd in jds:
        if "text" not in jd or "title" not in jd:
            continue
        jd_skills = extract_known_skills_from_text(jd["text"])
        resume_skills_set = extract_known_skills_from_text(resume_text)

        skill_overlap = len(jd_skills & resume_skills_set) / max(len(jd_skills), 1)
        jd_emb = bert_model.encode(jd["text"], convert_to_tensor=True)
        bert_score = util.pytorch_cos_sim(resume_emb, jd_emb).item()
        score = round(((0.7 * skill_overlap) + (0.3 * bert_score)) * 100, 2)        

        jd_skills = extract_known_skills_from_text(jd["text"])
        resume_skills = extract_known_skills_from_text(resume_text)
        missing = sorted(list(jd_skills - resume_skills))[:5]

        results.append({
            "title": jd["title"],
            "fit_score": score,
            "top_missing_skills": missing,
            "recommendation": get_feedback(score),
        })

    results.sort(key=lambda x: x["fit_score"], reverse=True)
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "total_jds_compared": len(results),
        "best_match": results[0]["title"] if results else None,
        "rankings": results,
    }


@app.get("/roles")
async def get_supported_roles():
    """Return all supported job roles."""
    return {"roles": list(ROLE_SKILLS.keys())}


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────
class SkillRequest(BaseModel):
    skills: List[str]

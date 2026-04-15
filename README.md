# SkillScan – AI Resume Skill Extractor

SkillScan is an AI-powered resume analysis tool that extracts skills from resumes and compares them with required job skills. The project uses Natural Language Processing (NLP) with FastAPI as the backend and a simple HTML/CSS frontend.

## Features

- Resume text analysis using NLP
- Skill extraction from resume content
- Skill matching against job requirements
- FastAPI backend for processing
- Simple and clean frontend interface
- REST API support
- Easy deployment and testing

## Tech Stack

**Backend**

- Python
- FastAPI
- Uvicorn
- NLP (spaCy / sklearn / pandas depending on what you used)

**Frontend**

- HTML
- CSS

**Tools**

- Git
- GitHub
- Virtual Environment (venv)

## Project Structure

```
SkillScan
│
├── nlp.py              # FastAPI backend
├── index.html          # Frontend UI
├── style.css           # Frontend styling
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
├── venv (ignored)
```

## How It Works

1. User uploads or inputs resume text
2. Backend processes text using NLP techniques
3. Skills are extracted and analyzed
4. Matching skills are returned as output
5. Results displayed on frontend

## Installation and Setup

### 1 Clone repository

```
git clone https://github.com/afsana158/SkillScan.git
cd SkillScan
```

### 2 Create virtual environment

```
python -m venv venv
```

### 3 Activate environment

Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

### 4 Install dependencies

```
pip install -r requirements.txt
```

### 5 Run backend server

```
python -m uvicorn nlp:app --reload
```

Backend runs on:

```
http://127.0.0.1:8000
```

API docs:

```
http://127.0.0.1:8000/docs
```

### 6 Run frontend

Open:

```
index.html
```

OR run:

```
python -m http.server 5500
```

Then open:

```
http://localhost:5500
```

## API Endpoints

Example endpoint:

```
POST /analyze
```

Returns extracted skills and matching score.

## Future Improvements

- Resume PDF upload support
- Better ML model for skill matching
- Authentication system
- Database integration
- Job recommendation system
- React frontend

## Learning Outcomes

This project helped in understanding:

- FastAPI backend development
- NLP basics
- API integration
- Full stack ML project structure
- Git workflow

## Author

**Soha Afsana**

GitHub:
https://github.com/afsana158

## License

This project is for educational purposes.

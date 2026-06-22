# Resume ATS Analyzer

Multi-agent system that scores a resume against a job description, finds keyword/skill gaps, and iteratively rewrites resume bullets to close them. Built with CrewAI agents orchestrated as a Flow, backed by the OpenAI API.

## How it works

1. **Parser** transcribes the resume and self-checks its own confidence, retrying once if the transcription looks unreliable.
2. **Scorer** rates the resume against the job description.
3. **Gap Analyst** finds missing keywords and skills, ranked by impact.
4. **Advisor** proposes grounded rewrites; **Critic** verifies each one against the real resume text before approving it.
5. **Strategist** decides each round whether to keep refining, retry the rewrites, or send it back to gap analysis — and whether that round's result is worth keeping.

Agents can also consult each other directly for clarification (bounded — one consult per pair per round, with cycle detection to prevent loops).

## Stack

- **Agents**: CrewAI, orchestrated as a `Flow` (not a fixed pipeline)
- **LLM**: OpenAI API (`gpt-4o-mini` by default)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Parsing**: `pdf2image` (PDFs, via vision model) + `python-docx` (DOCX)

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
MAX_OUTPUT_TOKENS=4096
```

Run the backend and frontend in separate terminals:

```bash
uvicorn app.api.routes:app --reload
streamlit run frontend/streamlit_app.py
```

## API

`POST /analyze` — multipart form with a `resume` file (PDF or DOCX) and a `job_description` string. Returns the parsed resume, match score, gap analysis, and suggested rewrites as JSON.

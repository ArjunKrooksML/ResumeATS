import os
import uuid

from fastapi import FastAPI, Form, UploadFile

from app.crew.ats_crew import analyze_resume
from app.debug_log import log_step
from app.models.schemas import AnalysisReport

app = FastAPI(title="Resume ATS Analyzer")

UPLOAD_DIR = "uploads"


def save_upload(file: UploadFile) -> str:
    extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{extension}")
    with open(file_path, "wb") as destination:
        destination.write(file.file.read())
    log_step(f"Saved upload '{file.filename}' to {file_path}")
    return file_path


@app.post("/analyze", response_model=AnalysisReport)
def analyze(resume: UploadFile, job_description: str = Form(...)):
    file_path = save_upload(resume)
    try:
        report = analyze_resume(file_path, job_description)
    finally:
        os.remove(file_path)
        log_step(f"Removed temporary file {file_path}")
    return report

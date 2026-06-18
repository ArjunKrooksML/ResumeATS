import requests
import streamlit as st

API_URL = "http://localhost:8000/analyze"


def show_score(score):
    st.metric("Match Score", f"{score['match_score']} / 100")
    st.write(score["reasoning"])
    st.write("Matched keywords:", ", ".join(score["matched_keywords"]))


def show_structural_risks(risks):
    if not risks:
        st.success("No structural risks detected.")
        return
    for risk in risks:
        st.warning(f"{risk['issue']} ({risk['severity']}): {risk['explanation']}")


def show_gaps(gaps):
    st.subheader("Missing Keywords")
    st.write(", ".join(gaps["missing_keywords"]))
    st.subheader("Priority Gaps")
    for gap in gaps["priority_gaps"]:
        st.write(f"- {gap}")


def show_suggestions(suggestions):
    for suggestion in suggestions:
        st.write(f"**Original:** {suggestion['original_bullet']}")
        st.write(f"**Revised:** {suggestion['revised_bullet']}")
        st.caption(f"Addresses: {suggestion['addresses_gap']}")


def show_report(report):
    show_score(report["score"])
    show_structural_risks(report["parsed_resume"]["structural_risks"])
    show_gaps(report["gaps"])
    st.subheader("Suggested Improvements")
    show_suggestions(report["final_suggestions"])
    st.caption(f"Refinement rounds used: {report['iterations_used']}")


st.title("Resume ATS Analyzer")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
job_description = st.text_area("Paste the job description")

if st.button("Analyze") and uploaded_file and job_description:
    with st.spinner("Analyzing resume..."):
        files = {"resume": (uploaded_file.name, uploaded_file.getvalue())}
        data = {"job_description": job_description}
        response = requests.post(API_URL, files=files, data=data)

    if response.status_code == 200:
        show_report(response.json())
    else:
        st.error(f"Analysis failed: {response.text}")

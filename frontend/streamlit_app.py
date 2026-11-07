import requests
import streamlit as st

API_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="ResumeATS", layout="wide", page_icon="📄")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { padding: 6px 18px; border-radius: 6px 6px 0 0; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
}
.covered   { background: #dcfce7; color: #15803d; }
.partial   { background: #fef9c3; color: #a16207; }
.missing   { background: #fee2e2; color: #b91c1c; }
.closed    { background: #dcfce7; color: #15803d; }
.attempted { background: #ffedd5; color: #c2410c; }
.screened  { background: #f1f5f9; color: #475569; }
.open      { background: #eff6ff; color: #1d4ed8; }
.high      { background: #fee2e2; color: #b91c1c; }
.medium    { background: #fef9c3; color: #a16207; }
.low       { background: #dcfce7; color: #15803d; }

.card {
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
}
.label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 3px;
}
.original-text { color: #64748b; font-size: 0.88rem; line-height: 1.5; }
.revised-text  { color: #0f172a; font-size: 0.88rem; font-weight: 500; line-height: 1.5; }
.gap-label     { margin-top: 8px; }
.red-flag-item { padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.88rem; }
.req-row       { display: flex; justify-content: space-between; align-items: flex-start;
                 padding: 8px 0; border-bottom: 1px solid #f1f5f9; gap: 12px; }
.req-name      { font-size: 0.88rem; flex: 1; }
.req-evidence  { font-size: 0.78rem; color: #64748b; flex: 2; font-style: italic; }
</style>
""", unsafe_allow_html=True)


def badge(label, kind):
    return f'<span class="badge {kind.lower()}">{label}</span>'


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ResumeATS")
st.caption("Multi-agent resume analysis — scores, critiques, rewrites, and fact-checks iteratively.")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    uploaded_file = st.file_uploader("Resume", type=["pdf", "docx"], label_visibility="collapsed",
                                     help="PDF or DOCX")
    st.caption("Upload a PDF or DOCX resume")
with c2:
    job_description = st.text_area("Job description", height=160, placeholder="Paste the full job description here…",
                                   label_visibility="collapsed")

ready = bool(uploaded_file and job_description and job_description.strip())
if st.button("Analyze", type="primary", disabled=not ready, use_container_width=False):
    with st.spinner("Running 6-agent pipeline — this takes a minute…"):
        try:
            resp = requests.post(
                API_URL,
                files={"resume": (uploaded_file.name, uploaded_file.getvalue())},
                data={"job_description": job_description},
                timeout=300,
            )
            if resp.ok:
                st.session_state.report = resp.json()
                st.session_state.error = None
            else:
                st.session_state.report = None
                st.session_state.error = resp.text
        except Exception as e:
            st.session_state.report = None
            st.session_state.error = str(e)

if st.session_state.get("error"):
    st.error(f"Analysis failed: {st.session_state.error}")

if not st.session_state.get("report"):
    st.stop()

report = st.session_state.report
score  = report["score"]
gaps   = report["gaps"]
jd     = report.get("jd_analysis") or {}
clos   = report.get("gap_closures", [])
cov    = report.get("coverage", [])
suggs  = report.get("final_suggestions", [])
risks  = report["parsed_resume"].get("structural_risks", [])
iters  = report.get("iterations_used", 0)

n_closed   = sum(1 for g in clos if g["status"] == "closed")
n_gaps     = len(gaps["priority_gaps"])

# ── Summary strip ─────────────────────────────────────────────────────────────
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Match Score",    f"{score['match_score']} / 100")
m2.metric("JD Credibility", f"{jd.get('credibility_score', '—')} / 100" if jd else "—")
m3.metric("Priority Gaps",  n_gaps)
m4.metric("Gaps Closed",    f"{n_closed} / {len(clos)}" if clos else str(n_closed))
m5.metric("Rounds Used",    iters)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_coverage, tab_gaps, tab_suggestions, tab_risks = st.tabs([
    "Overview", "Coverage Matrix", "Gap Closures", "Suggestions", "ATS Risks"
])


# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Score breakdown")
        st.progress(score["match_score"] / 100)
        st.caption(score["reasoning"])

        st.markdown("**Matched keywords**")
        kws = score.get("matched_keywords", [])
        if kws:
            st.markdown(" ".join(f'<span class="badge covered">{k}</span>' for k in kws),
                        unsafe_allow_html=True)
        else:
            st.caption("None detected.")

    with right:
        st.subheader("JD quality")
        if jd:
            cred = jd.get("credibility_score", 0)
            color = "#15803d" if cred >= 70 else "#a16207" if cred >= 40 else "#b91c1c"
            st.markdown(
                f'<div style="font-size:2.4rem;font-weight:700;color:{color};">{cred}<span style="font-size:1rem;color:#94a3b8;"> / 100</span></div>',
                unsafe_allow_html=True,
            )
            if not jd.get("is_analyzable", True):
                st.warning("JD is too vague for meaningful gap analysis.")
            st.caption(jd.get("summary", ""))
            flags = jd.get("red_flags", [])
            if flags:
                st.markdown("**Red flags**")
                for f in flags:
                    st.markdown(f'<div class="red-flag-item">⚠ {f}</div>', unsafe_allow_html=True)
            else:
                st.success("No red flags detected.")
        else:
            st.caption("JD analysis not available.")


# ── Tab 2: Coverage Matrix ────────────────────────────────────────────────────
with tab_coverage:
    if not cov:
        st.info("Coverage data not available — rerun the analysis to generate it.")
    else:
        covered_n  = sum(1 for r in cov if r["status"] == "covered")
        partial_n  = sum(1 for r in cov if r["status"] == "partial")
        missing_n  = sum(1 for r in cov if r["status"] == "missing")

        c1, c2, c3 = st.columns(3)
        c1.metric("Covered",  covered_n)
        c2.metric("Partial",  partial_n)
        c3.metric("Missing",  missing_n)
        st.divider()

        for item in sorted(cov, key=lambda x: {"covered": 0, "partial": 1, "missing": 2}.get(x["status"], 3)):
            evidence = item.get("covered_by", "").strip()
            evidence_html = f'<span class="req-evidence">{evidence}</span>' if evidence else ""
            st.markdown(
                f'<div class="req-row">'
                f'<span class="req-name">{item["requirement"]}</span>'
                f'{badge(item["status"], item["status"])}'
                f'{evidence_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Tab 3: Gap Closures ───────────────────────────────────────────────────────
with tab_gaps:
    if not clos:
        st.info("No gap lifecycle data available.")
    else:
        STATUS_ICON = {"closed": "✅", "attempted": "⚠️", "screened": "⏭", "open": "○"}
        order = {"closed": 0, "attempted": 1, "open": 2, "screened": 3}
        for g in sorted(clos, key=lambda x: order.get(x["status"], 4)):
            status = g["status"]
            icon = STATUS_ICON.get(status, "○")
            with st.expander(f"{icon}  {g['gap']}", expanded=(status == "closed")):
                if status == "closed":
                    st.markdown(f"**Closed in round {g.get('closed_in_round', '?')}**")
                    if g.get("closed_by"):
                        st.markdown(f'<div class="label">Revised bullet</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="revised-text">{g["closed_by"]}</div>', unsafe_allow_html=True)
                elif status == "attempted":
                    st.markdown("Advisor attempted rewrites but Critic rejected all — no grounded suggestion found.")
                elif status == "screened":
                    st.markdown("Screened out before the Advisor ran — resume had no relevant experience to reframe.")
                else:
                    st.markdown("Not addressed in this run.")


# ── Tab 4: Suggestions ────────────────────────────────────────────────────────
with tab_suggestions:
    if not suggs:
        st.info("No suggestions were approved this run.")
    else:
        for s in suggs:
            st.markdown(
                f'<div class="card">'
                f'<div class="label">Original</div>'
                f'<div class="original-text">{s["original_bullet"]}</div>'
                f'<div style="margin:10px 0;border-top:1px dashed #e2e8f0;"></div>'
                f'<div class="label">Revised</div>'
                f'<div class="revised-text">{s["revised_bullet"]}</div>'
                f'<div class="gap-label">{badge(s["addresses_gap"], "closed")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Tab 5: ATS Risks ──────────────────────────────────────────────────────────
with tab_risks:
    parsed = report["parsed_resume"]
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Skills detected",   len(parsed.get("skills", [])))
    rc2.metric("Years experience",  parsed.get("years_experience", "—"))
    rc3.metric("Structural risks",  len(risks))

    if parsed.get("skills"):
        st.markdown("**Detected skills**")
        st.markdown(
            " ".join(f'<span class="badge covered">{s}</span>' for s in parsed["skills"]),
            unsafe_allow_html=True,
        )

    st.divider()
    SEV_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    if risks:
        for r in risks:
            sev = r.get("severity", "medium")
            icon = SEV_ICON.get(sev, "🟡")
            with st.expander(f'{icon}  {r["issue"]}'):
                st.write(r["explanation"])
    else:
        st.success("No structural ATS risks detected.")

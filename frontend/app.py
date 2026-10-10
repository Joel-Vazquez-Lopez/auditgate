
import html
import os

import requests
import streamlit as st

API_URL = os.getenv("AUDITGATE_API_URL", "http://127.0.0.1:3000")

st.set_page_config(
    page_title="AuditGate | Evidence Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: #0B1020;
        color: #E8EDF7;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background: #11182A;
        border-right: 1px solid #263149;
    }

    h1, h2, h3 {
        color: #F5F7FC !important;
        letter-spacing: -0.03em;
    }

    .hero {
        background: linear-gradient(115deg, #162A48, #11192B 65%, #102F39);
        border: 1px solid #2B405C;
        border-radius: 20px;
        padding: 30px 34px;
        margin-bottom: 26px;
    }

    .hero-label {
        color: #56DBD0;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1.8px;
        margin-bottom: 8px;
        color: white;
    }

    .hero-description {
        color: #B3C0D5;
        font-size: 16px;
        line-height: 1.7;
        max-width: 780px;
    }

    .verdict {
        border-radius: 16px;
        padding: 22px 26px;
        margin: 20px 0;
        border: 1px solid;
    }

    .verdict-ALLOW {
        background: #102D2A;
        border-color: #256B59;
    }

    .verdict-BLOCK {
        background: #341C27;
        border-color: #A0445B;
    }

    .verdict-REVIEW {
        background: #30291B;
        border-color: #97732B;
    }

    .verdict-NO_CHECK_NEEDED {
        background: #20283A;
        border-color: #465B80;
    }

    .verdict-title {
        font-size: 27px;
        font-weight: 800;
        margin-bottom: 5px;
        color: #F5F7FC;
    }

    .verdict-description {
        color: #C4CEDD;
        font-size: 14px;
    }

    .section-label {
        color: #64D8D0;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin: 24px 0 10px;
    }

    div[data-testid="stMetric"] {
        background: #151E31;
        border: 1px solid #2A3850;
        border-radius: 14px;
        padding: 14px 18px;
    }

    div.stButton > button[kind="primary"] {
        background: #34BFAF;
        color: #071421;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #55E2D1;
        color: #071421;
    }

    .footer-note {
        color: #8594AA;
        font-size: 12px;
        margin-top: 35px;
    }
</style>
""", unsafe_allow_html=True)


def safe(value):
    return html.escape(str(value or ""))


def status_icon(status):
    return {
        "ALLOW": "✓",
        "BLOCK": "✕",
        "REVIEW": "!",
        "NO_CHECK_NEEDED": "—",
    }.get(status, "?")


def show_result(result):
    decision = result.get("decision", "UNKNOWN")
    reason = result.get("reason", "")
    claims = result.get("claims") or []

    css_status = decision if decision in {
        "ALLOW", "BLOCK", "REVIEW", "NO_CHECK_NEEDED"
    } else "NO_CHECK_NEEDED"

    st.markdown('<div class="section-label">AUDIT REPORT</div>',
                unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="verdict verdict-{css_status}">
            <div class="verdict-title">
                {status_icon(decision)} {safe(decision)}
            </div>
            <div class="verdict-description">{safe(reason)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    statuses = ["ALLOW", "REVIEW", "BLOCK", "NO_CHECK_NEEDED"]
    labels = ["Allowed", "Needs review", "Blocked", "Not verifiable"]

    cols = st.columns(4)
    for col, status, label in zip(cols, statuses, labels):
        count = sum(c.get("decision") == status for c in claims)
        col.metric(label, count)

    st.markdown('<div class="section-label">CLAIM INTELLIGENCE</div>',
                unsafe_allow_html=True)

    if not claims:
        st.info("No claims were returned.")
        return

    for index, claim in enumerate(claims, 1):
        status = claim.get("decision", "UNKNOWN")
        claim_text = claim.get("claim", "")

        heading = (
            f"{status_icon(status)}  {index:02d}  "
            f"{status}  ·  {claim_text}"
        )

        with st.expander(heading, expanded=index == 1):
            st.write(claim.get("reason", ""))

            provenance = []
            if claim.get("page") is not None:
                provenance.append(f"Page {claim['page']}")
            if claim.get("paragraph") is not None:
                provenance.append(f"Paragraph {claim['paragraph']}")
            if claim.get("section"):
                provenance.append(f"Section: {claim['section']}")

            if provenance:
                st.caption("  ·  ".join(provenance))

            evidence = claim.get("evidence") or []
            trace = claim.get("tacer_trace") or []

            tab_evidence, tab_trace = st.tabs([
                f"Evidence ({len(evidence)})",
                f"TACER trace ({len(trace)})",
            ])

            with tab_evidence:
                if not evidence:
                    st.info("No evidence required or retrieved.")

                for item in evidence:
                    title = item.get("source_title") or "Source"
                    url = item.get("source_url") or ""
                    label = item.get("verification_label", "unknown")
                    confidence = item.get("verification_confidence")

                    with st.container(border=True):
                        if url.startswith(("http://", "https://")):
                            st.markdown(f"**[{safe(title)}]({url})**")
                        else:
                            st.write(title)

                        st.write(item.get("text", ""))

                        metadata = f"Evidence assessment: {label}"
                        if isinstance(confidence, (int, float)):
                            metadata += f" · Confidence: {confidence:.1%}"

                        st.caption(metadata)

            with tab_trace:
                if not trace:
                    st.info("No additional evidence-acquisition steps.")

                for step in trace:
                    iteration = step.get("iteration", "?")
                    action = step.get("action", "Unknown")
                    with st.container(border=True):
                        st.markdown(f"**Step {iteration}: {action}**")
                        st.caption(
                            f"Candidates: {step.get('candidate_count', '—')} "
                            f"· Coverage: {step.get('claim_coverage', 0):.2f} "
                            f"· Source diversity: {step.get('source_diversity', 0):.2f}"
                        )

    st.download_button(
        "⬇ Download audit report (JSON)",
        data=__import__("json").dumps(result, indent=2),
        file_name="auditgate_report.json",
        mime="application/json",
    )


def submit_audit(endpoint, **kwargs):
    try:
        with st.spinner("AuditGate is gathering and evaluating evidence..."):
            response = requests.post(
                f"{API_URL}{endpoint}",
                timeout=600,
                **kwargs,
            )
            response.raise_for_status()
            st.session_state["audit_result"] = response.json()
    except requests.RequestException as exc:
        st.error(f"Audit request failed: {exc}")


with st.sidebar:
    st.markdown("## 🛡️ AuditGate")
    st.caption("EVIDENCE INTELLIGENCE")
    st.divider()

    st.markdown("**SYSTEM PIPELINE**")
    st.write("01 · Claim extraction")
    st.write("02 · Verifiability filtering")
    st.write("03 · Evidence retrieval")
    st.write("04 · Neural verification")
    st.write("05 · TACER acquisition")
    st.write("06 · Decision aggregation")

    st.divider()
    st.markdown("**API CONNECTION**")

    try:
        health = requests.get(f"{API_URL}/health/live", timeout=2)
        if health.ok:
            st.success("Rust API connected")
        else:
            st.warning("Rust API returned an error")
    except requests.RequestException:
        st.error("Rust API offline")

    st.caption("Local MVP · Research prototype")


st.markdown(
    """
    <div class="hero">
        <div class="hero-label">EVIDENCE-GROUNDED AI AUDITING</div>
        <div class="hero-title">AuditGate</div>
        <div class="hero-description">
            Don't just generate answers. Verify them.
            AuditGate breaks content into atomic claims, investigates
            evidence from real sources, and explains whether each
            claim is safe to accept, needs review, or should be blocked.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">NEW AUDIT</div>',
            unsafe_allow_html=True)

tab_text, tab_upload = st.tabs([
    "✍️ Paste text",
    "📄 Upload document",
])

with tab_text:
    st.write("Analyze a passage, AI-generated response, or article.")

    text = st.text_area(
        "Content",
        height=220,
        placeholder=(
            "Apollo 11 landed humans on the Moon in 1969. "
            "Mars is the largest planet in the Solar System."
        ),
    )

    if st.button("Run text audit →", type="primary"):
        if text.strip():
            submit_audit("/v1/audit-text", json={"text": text})
        else:
            st.warning("Enter some text first.")

with tab_upload:
    st.write("Upload a document to extract and verify its factual claims.")

    uploaded = st.file_uploader(
        "Choose a file",
        type=["txt", "md", "pdf", "docx"],
        max_upload_size=10,
    )

    if st.button("Run document audit →", type="primary"):
        if uploaded is None:
            st.warning("Choose a document first.")
        else:
            submit_audit(
                "/v1/audit-document",
                files={"file": (uploaded.name, uploaded.getvalue())},
            )

if "audit_result" in st.session_state:
    st.divider()
    show_result(st.session_state["audit_result"])

st.markdown(
    '<div class="footer-note">'
    'AuditGate is an experimental evidence-grounded decision-support '
    'system. Model confidence is not a guarantee of factual correctness.'
    '</div>',
    unsafe_allow_html=True,
)

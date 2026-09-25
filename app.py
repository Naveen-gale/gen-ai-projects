import html
import time
import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import (
    build_search_agent, build_reader_agent,
    invoke_search_agent, invoke_reader_agent,
    invoke_writer_chain, invoke_critic_chain,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STEP_DELAY = 4  # seconds between steps

st.set_page_config(
    page_title="ResearchMind - AI Research Agent",
    page_icon="magnifying_glass_tilted_right",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: "DM Sans", sans-serif; color: #e8e4dc; }
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }
.hero { text-align: center; padding: 3.5rem 0 2.5rem; }
.hero h1 { font-family: "Syne", sans-serif; font-size: 4rem; font-weight: 800; color: #f0ebe0; margin: 0 0 1rem; }
.hero h1 span { color: #ff8c32; }
.hero-sub { font-size: 1.05rem; color: #a09890; max-width: 520px; margin: 0 auto; line-height: 1.65; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent); margin: 2rem 0; }
.step-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem; position: relative; }
.step-card.active { border-color: rgba(255,140,50,0.4); background: rgba(255,140,50,0.04); }
.step-card.done   { border-color: rgba(80,200,120,0.3); background: rgba(80,200,120,0.03); }
.step-card::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; border-radius:14px 0 0 14px; background:rgba(255,255,255,0.05); }
.step-card.active::before { background: #ff8c32; }
.step-card.done::before   { background: #50c878; }
.step-header { display:flex; align-items:center; gap:0.8rem; margin-bottom:0.3rem; }
.step-num   { font-family:"DM Mono",monospace; font-size:0.68rem; color:#ff8c32; opacity:0.7; }
.step-title { font-family:"Syne",sans-serif; font-size:0.95rem; font-weight:700; color:#f0ebe0; }
.step-status { margin-left:auto; font-family:"DM Mono",monospace; font-size:0.68rem; }
.status-waiting { color:#555; } .status-running { color:#ff8c32; } .status-done { color:#50c878; }
.section-heading { font-family:"Syne",sans-serif; font-size:1.3rem; font-weight:700; color:#f0ebe0; margin:2rem 0 1rem; }
.report-panel { background:rgba(255,255,255,0.025); border:1px solid rgba(255,140,50,0.2); border-radius:16px; padding:2rem 2.5rem; margin-top:1rem; }
.feedback-panel { background:rgba(255,255,255,0.025); border:1px solid rgba(80,200,120,0.2); border-radius:16px; padding:2rem 2.5rem; margin-top:1rem; }
.panel-label { font-family:"DM Mono",monospace; font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:1.2rem; padding-bottom:0.7rem; }
.panel-label.orange { color:#ff8c32; border-bottom:1px solid rgba(255,140,50,0.15); }
.panel-label.green  { color:#50c878; border-bottom:1px solid rgba(80,200,120,0.15); }
.stButton > button { background: linear-gradient(135deg,#ff8c32 0%,#ff5a1a 100%) !important; color:#0a0a0f !important; font-weight:700 !important; border:none !important; border-radius:10px !important; box-shadow:0 4px 20px rgba(255,140,50,0.3) !important; width:100%; }
.notice { font-family:"DM Mono",monospace; font-size:0.72rem; color:#605850; text-align:center; margin-top:3rem; }
</style>
""", unsafe_allow_html=True)


def step_card(num, title, state, desc=""):
    status_map = {
        "waiting": ("WAITING",  "status-waiting"),
        "running": ("RUNNING",  "status-running"),
        "done":    ("DONE",     "status-done"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    desc_html = f"<div style='font-size:0.82rem;color:#706860;margin-top:0.3rem;'>{html.escape(desc)}</div>" if desc else ""
    st.markdown(f"""
    <div class="step-card {card_cls}">
      <div class="step-header">
        <span class="step-num">{html.escape(num)}</span>
        <span class="step-title">{html.escape(title)}</span>
        <span class="step-status {cls}">{label}</span>
      </div>
      {desc_html}
    </div>""", unsafe_allow_html=True)


for key in ("results", "running", "done", "error"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "results" else False

# Detect which LLM is active
_google_key = os.getenv("GOOGLE_API_KEY", "")
_mistral_key = os.getenv("MISTRALAI_API_KEY", "")
if _google_key and _google_key != "your_google_api_key_here":
    _llm_badge = "Google Gemini 1.5 Flash"
    _llm_color = "#4285F4"
elif _mistral_key:
    _llm_badge = "Mistral open-mistral-nemo"
    _llm_color = "#ff8c32"
else:
    _llm_badge = "No API key set"
    _llm_color = "#ff4444"

st.markdown(f"""
<div class="hero">
  <h1>Research<span>Mind</span></h1>
  <p class="hero-sub">Four AI agents collaborate: search, scrape, write, critique.</p>
  <p style="margin-top:0.8rem;font-size:0.78rem;font-family:'DM Mono',monospace;color:{_llm_color};">
    LLM: {_llm_badge}
  </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

col_input, _, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    topic = st.text_input("Research Topic", placeholder="e.g. Quantum computing 2025", key="topic_input")
    run_btn = st.button("Run Research Pipeline", use_container_width=True)

    if _google_key == "your_google_api_key_here" or not _google_key:
        st.warning(
            "**Add a Google API key to remove rate limits.**\n\n"
            "1. Go to https://aistudio.google.com/app/apikey\n"
            "2. Click **Create API key** (free)\n"
            "3. Paste it as `GOOGLE_API_KEY=...` in your `.env` file\n"
            "4. Restart the app"
        )

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)
    r = st.session_state.results

    def _state(step):
        steps = ["search", "reader", "writer", "critic"]
        if not r:
            return "waiting"
        if step in r:
            return "done"
        if st.session_state.running:
            for k in steps:
                if k not in r:
                    return "running" if k == step else "waiting"
        return "waiting"

    step_card("01", "Search Agent",  _state("search"), "Gathers recent web information")
    step_card("02", "Reader Agent",  _state("reader"), "Scrapes & extracts deep content")
    step_card("03", "Writer Chain",  _state("writer"), "Drafts the full research report")
    step_card("04", "Critic Chain",  _state("critic"), "Reviews & scores the report")

if st.session_state.error:
    err = str(st.session_state.error)
    st.error(f"An error occurred: {err}")
    if "429" in err or "rate" in err.lower() or "quota" in err.lower():
        st.info(
            "**Rate limit hit on Mistral free tier.**\n\n"
            "Permanent fix: add a free Google Gemini API key to your `.env`:\n"
            "1. Visit https://aistudio.google.com/app/apikey\n"
            "2. Create a free key\n"
            "3. Add `GOOGLE_API_KEY=your_key` to `.env`\n"
            "4. Restart: `streamlit run app.py`\n\n"
            "Gemini gives **15 req/min and 1M tokens/day** for free."
        )
    st.session_state.error = False

if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.error = False
        st.rerun()

if st.session_state.running and not st.session_state.done:
    results = {}
    topic_val = st.session_state.topic_input
    try:
        with st.spinner("Step 1/4 - Search Agent working..."):
            sa = build_search_agent()
            sr = invoke_search_agent(sa, topic_val)
            results["search"] = sr["messages"][-1].content
            st.session_state.results = dict(results)
        time.sleep(STEP_DELAY)

        with st.spinner("Step 2/4 - Reader Agent scraping..."):
            ra = build_reader_agent()
            rr = invoke_reader_agent(ra, topic_val, results["search"])
            results["reader"] = rr["messages"][-1].content
            st.session_state.results = dict(results)
        time.sleep(STEP_DELAY)

        with st.spinner("Step 3/4 - Writer drafting report..."):
            research_combined = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"SCRAPED CONTENT:\n{results['reader']}"
            )
            results["writer"] = invoke_writer_chain(topic_val, research_combined)
            st.session_state.results = dict(results)
        time.sleep(STEP_DELAY)

        with st.spinner("Step 4/4 - Critic reviewing..."):
            results["critic"] = invoke_critic_chain(results["writer"])
            st.session_state.results = dict(results)

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        st.session_state.error = str(exc)
        st.session_state.results = dict(results)
    finally:
        st.session_state.running = False
        st.session_state.done = True
        st.rerun()

r = st.session_state.results
if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    if "search" in r:
        with st.expander("Search Results (raw)", expanded=False):
            st.text(r["search"])

    if "reader" in r:
        with st.expander("Scraped Content (raw)", expanded=False):
            st.text(r["reader"])

    if "writer" in r:
        st.markdown('<div class="report-panel"><div class="panel-label orange">Final Research Report</div></div>', unsafe_allow_html=True)
        st.markdown(r["writer"])
        st.download_button(
            label="Download Report (.md)",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in r:
        st.markdown('<div class="feedback-panel"><div class="panel-label green">Critic Feedback</div></div>', unsafe_allow_html=True)
        st.markdown(r["critic"])

st.markdown('<div class="notice">ResearchMind - Powered by LangChain multi-agent pipeline - Built with Streamlit</div>', unsafe_allow_html=True)
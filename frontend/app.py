from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -------------------------------------------------------------------
# App configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="TCS | Financial Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "financial_metrics"
API_BASE = os.getenv(
    "API_BASE_URL",
    "https://financial-intelligence-agent-pmgy.onrender.com",
).rstrip("/")
API_URL = f"{API_BASE}/ask"
HEALTH_URL = f"{API_BASE}/ready"

YEARS = ["2022-03-31", "2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"]
YEAR_LABELS = {year: "FY" + year[:4] for year in YEARS}


# -------------------------------------------------------------------
# Visual system
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --ink: #eaf1f7;
      --muted: #91a3b5;
      --panel: #111d2b;
      --panel-2: #162638;
      --line: rgba(148, 178, 204, .16);
      --cyan: #53d6df;
      --mint: #9be4c1;
      --gold: #f1c879;
    }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp {
      background:
        radial-gradient(ellipse at 8% 0%, rgba(36, 105, 126, .20), transparent 34%),
        radial-gradient(ellipse at 92% 8%, rgba(44, 83, 123, .16), transparent 30%),
        #09131f;
      color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(9,19,31,.72); }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #101d2b 0%, #0c1724 100%);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { letter-spacing: -.035em; color: #f1f6fb; }
    p, li { color: #c1cfdb; }
    .eyebrow {
      color: var(--cyan); text-transform: uppercase; letter-spacing: .18em;
      font-size: .68rem; font-weight: 800; margin-bottom: .55rem;
    }
    .hero-title {
      font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.06;
      font-weight: 800; letter-spacing: -.06em; color: #f5f8fb; margin: 0;
    }
    .hero-subtitle { color: #9fb2c4; font-size: .98rem; margin-top: .7rem; max-width: 760px; }
    .hero-meta { color: #7890a5; font-family: 'DM Mono', monospace; font-size: .72rem; margin-top: 1.2rem; }
    .status-pill {
      display: inline-flex; align-items: center; gap: .45rem; border: 1px solid rgba(83,214,223,.28);
      color: #9ce8e7; background: rgba(83,214,223,.08); border-radius: 999px;
      padding: .42rem .75rem; font-size: .72rem; font-weight: 700; letter-spacing: .04em;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #71dfb3; display: inline-block; }
    .section-kicker { color: var(--cyan); font-size: .68rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .35rem; }
    .section-title { font-size: 1.45rem; font-weight: 800; color: #edf4f9; letter-spacing: -.04em; margin-bottom: .2rem; }
    .section-note { color: #91a3b5; font-size: .84rem; margin-bottom: 1rem; }
    .metric-card {
      background: linear-gradient(145deg, rgba(23,42,59,.96), rgba(15,29,43,.96));
      border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.1rem;
      min-height: 118px; box-shadow: 0 10px 28px rgba(0,0,0,.12);
    }
    .metric-label { color: #91a8bb; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .metric-value { color: #f1f7fb; font-size: 1.72rem; font-weight: 800; letter-spacing: -.05em; margin-top: .35rem; }
    .metric-foot { color: #89a1b5; font-size: .72rem; margin-top: .2rem; }
    .panel {
      background: rgba(17,29,43,.82); border: 1px solid var(--line);
      border-radius: 18px; padding: 1.15rem 1.25rem; margin-bottom: .9rem;
    }
    .panel-title { color: #e8f0f6; font-weight: 800; font-size: 1rem; }
    .panel-copy { color: #91a5b7; font-size: .82rem; line-height: 1.55; }
    .source-chip {
      display: inline-block; border: 1px solid rgba(83,214,223,.25); color: #a4e6e4;
      background: rgba(83,214,223,.07); border-radius: 7px; padding: .2rem .5rem;
      font-family: 'DM Mono', monospace; font-size: .7rem; margin: .15rem .25rem .15rem 0;
    }
    /* Real radio navigation rendered as separate button-like pills. */
    div[data-testid="stRadio"] > label { display: none !important; }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
      display: flex !important;
      flex-wrap: wrap !important;
      gap: .55rem !important;
      align-items: center !important;
    }
    div[data-testid="stRadio"] label {
      display: inline-flex !important;
      align-items: center !important;
      width: auto !important;
      min-height: 2.75rem !important;
      padding: .62rem .9rem !important;
      margin: 0 !important;
      border: 1px solid rgba(120,160,190,.3) !important;
      border-radius: 12px !important;
      background: #142638 !important;
      color: #b7c8d6 !important;
      cursor: pointer !important;
      box-shadow: 0 3px 10px rgba(0,0,0,.12) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
      border-color: #53d6df !important;
      background: linear-gradient(145deg, #21465a, #173247) !important;
      color: #aaf1ed !important;
      box-shadow: 0 0 0 1px rgba(83,214,223,.16), 0 6px 16px rgba(0,0,0,.2) !important;
    }
    div[data-testid="stRadio"] label > div:first-child { display: none !important; }
    div[data-testid="stRadio"] label p { color: inherit !important; font-weight: 700 !important; margin: 0 !important; }
    div[data-testid="stButton"] button {
      border: 1px solid rgba(120,160,190,.22); background: #142638; color: #dce9f2;
      border-radius: 10px; font-weight: 650; transition: all .18s ease;
    }
    div[data-testid="stButton"] button:hover { border-color: #53d6df; color: #9ce8e7; background: #1a3347; }
    div[data-testid="stChatInput"] textarea { background: #142638; }
    div[data-testid="stChatMessage"] { background: rgba(19,35,51,.72); border: 1px solid var(--line); border-radius: 15px; padding: .9rem; }
    div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: rgba(17,29,43,.65); }

    /* Responsive layout: preserve readable cards and charts on phones/tablets. */
    @media (max-width: 1100px) {
      .block-container { padding-left: 1.35rem; padding-right: 1.35rem; }
      div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-wrap: wrap !important;
        overflow: visible !important;
        gap: .45rem !important;
        padding-bottom: .5rem;
      }
      div[data-testid="stRadio"] label { flex: 0 1 auto !important; min-width: max-content !important; }
      .metric-value { font-size: clamp(1.2rem, 3vw, 1.65rem); overflow-wrap: anywhere; }
    }
    @media (max-width: 760px) {
      .block-container {
        padding: 1.1rem 1rem 2.2rem;
        max-width: 100%;
      }
      .hero-title { font-size: clamp(1.85rem, 8vw, 2.45rem); line-height: 1.12; }
      .hero-subtitle { font-size: .9rem; line-height: 1.6; }
      .hero-meta { font-size: .62rem; line-height: 1.7; }
      .section-title { font-size: 1.3rem; }
      .metric-card { min-height: 108px; padding: .9rem; }
      .metric-value { font-size: clamp(1.15rem, 5vw, 1.5rem); }
      .panel { padding: 1rem; }
      div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .75rem; }
      div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
      }
      div[data-testid="stRadio"] > div[role="radiogroup"] { gap: .4rem !important; }
      div[data-testid="stRadio"] label {
        font-size: .78rem !important;
        padding: .55rem .72rem !important;
        min-height: 2.6rem !important;
      }
      div[data-testid="stButton"] button {
        min-height: 2.8rem;
        white-space: normal;
        line-height: 1.35;
        padding: .65rem .8rem;
      }
      [data-testid="stSidebar"] { min-width: min(85vw, 320px); }
      [data-testid="stPlotlyChart"] { width: 100%; }
    }
    @media (max-width: 420px) {
      .block-container { padding-left: .75rem; padding-right: .75rem; }
      .hero-title { font-size: 1.8rem; }
      .metric-label { font-size: .66rem; }
      .metric-value { font-size: 1.18rem; }
      .metric-foot { font-size: .67rem; }
    }
    hr { border-color: var(--line); }
    .footer { color: #71869a; font-size: .72rem; text-align: center; padding-top: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Data access and formatting
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    income = pd.read_csv(DATA_DIR / "income_statement.csv", index_col=0)
    balance = pd.read_csv(DATA_DIR / "balance_sheet.csv", index_col=0)
    cash = pd.read_csv(DATA_DIR / "cash_flow.csv", index_col=0)
    ratios_path = DATA_DIR / "key_ratios.csv"
    ratios = pd.read_csv(ratios_path) if ratios_path.exists() else pd.DataFrame()
    for frame in (income, balance, cash):
        frame.columns = [str(c) for c in frame.columns]
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return income, balance, cash, ratios


def metric_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.index:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(frame.loc[name], errors="coerce")
    return values.dropna()


def latest_value(frame: pd.DataFrame, name: str) -> tuple[float | None, str | None]:
    values = metric_series(frame, name)
    if values.empty:
        return None, None
    dated = [(str(k), float(v)) for k, v in values.items() if str(k)[:4].isdigit()]
    if not dated:
        return None, None
    year, value = max(dated, key=lambda item: item[0])
    return value, year


def crore(value: float | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"₹{value / 1e7:,.{decimals}f} Cr"


def growth_text(frame: pd.DataFrame, name: str) -> str:
    values = metric_series(frame, name)
    dated = [(str(k), float(v)) for k, v in values.items() if str(k)[:4].isdigit()]
    dated.sort(key=lambda item: item[0])
    if len(dated) < 2 or dated[-2][1] == 0:
        return "Year-on-year change unavailable"
    change = (dated[-1][1] / dated[-2][1] - 1) * 100
    return f"{change:+.1f}% vs previous FY"


def chart_frame(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in names:
        values = metric_series(frame, name)
        for year, value in values.items():
            if str(year)[:4].isdigit():
                rows.append({"Financial year": "FY" + str(year)[:4], "Metric": name, "₹ Crore": value / 1e7})
    return pd.DataFrame(rows)


def style_chart(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=38, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#b9c9d6", size=11),
        title_font=dict(size=14, color="#eaf2f8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, linecolor="rgba(148,178,204,.15)"),
        yaxis=dict(gridcolor="rgba(148,178,204,.12)", zerolinecolor="rgba(148,178,204,.12)"),
    )
    return fig


def metric_card(label: str, value: str, foot: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def section_heading(kicker: str, title: str, note: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div>'
        f'<div class="section-note">{note}</div>',
        unsafe_allow_html=True,
    )


def set_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def set_dashboard_section(section: str) -> None:
    st.session_state.active_dashboard_section = section


# -------------------------------------------------------------------
# API health and question answering
# -------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def backend_health() -> tuple[bool, str]:
    """Check API readiness, including local retrieval indexes and Groq configuration.

    A short-lived timeout during a long AI generation must not immediately
    label the whole service offline, so retry once and report an unknown /
    delayed state rather than a definitive outage.
    """
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.get(HEALTH_URL, timeout=15)
            payload = response.json()
            if isinstance(payload, dict) and payload.get("status") == "not_ready":
                failed = [name for name, item in payload.get("checks", {}).items()
                          if isinstance(item, dict) and item.get("ready") is False]
                return False, "Backend not ready: " + ", ".join(failed or ["dependency check failed"])
            response.raise_for_status()
            if isinstance(payload, dict) and payload.get("status") == "ready":
                return True, "Backend and retrieval index ready"
            return False, "Unexpected readiness response"
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 0:
                import time
                time.sleep(0.6)

    if isinstance(last_error, requests.Timeout):
        return False, "Health check timed out twice (15s each). Render may be waking up or the request is not reaching the service."
    if isinstance(last_error, requests.HTTPError):
        status = last_error.response.status_code if last_error.response is not None else "unknown"
        return False, f"Health endpoint returned HTTP {status}."
    if isinstance(last_error, requests.ConnectionError):
        return False, "Could not connect to the Render host from Streamlit Cloud."
    return False, f"Health check failed ({type(last_error).__name__})."


@st.cache_resource(show_spinner=False)
def load_query_embedding_model():
    """Load the same BGE model used by the offline document-index build."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


def ask_question(question: str) -> dict[str, Any]:
    model = load_query_embedding_model()
    query_embedding = model.encode(
        "Represent this sentence for searching relevant passages: " + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    response = requests.post(
        API_URL,
        json={"question": question, "query_embedding": query_embedding},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("The research API returned an unexpected response.")
    return payload



@st.cache_data(ttl=60, show_spinner=False)
def load_market_quote(ticker: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE}/market/quote", params={"ticker": ticker}, timeout=25)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def load_market_history(ticker: str, period: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE}/market/history",
        params={"ticker": ticker, "period": period},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def load_financial_news(query: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE}/news", params={"q": query, "limit": 10}, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400, show_spinner=False)
def load_daily_sentiment(query: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE}/sentiment/daily",
        params={"q": query, "limit": 12},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_sources(sources: list[Any]) -> None:
    if not sources:
        st.caption("No source records were returned for this answer.")
        return
    with st.expander(f"Evidence trail · {len(sources)} source(s)", expanded=False):
        for index, source in enumerate(sources, start=1):
            if isinstance(source, dict):
                source_name = Path(str(source.get("source", "Unknown source"))).name
                page = source.get("page")
                chunk = source.get("chunk_id")
                methods = source.get("retrieval_methods") or source.get("methods") or []
                details = []
                if page:
                    details.append(f"page {page}")
                if chunk:
                    details.append(f"chunk {chunk}")
                st.markdown(f"**{index:02d} · {source_name}**")
                if details:
                    st.caption(" · ".join(details))
                if methods:
                    st.caption("Retrieved via: " + ", ".join(map(str, methods)))
            else:
                st.markdown(f"**{index:02d} · {Path(str(source)).name}**")


# -------------------------------------------------------------------
# Load financial datasets
# -------------------------------------------------------------------
try:
    income_df, balance_df, cash_df, ratio_df = load_data()
except Exception as exc:
    st.error(f"Financial datasets could not be loaded: {exc}")
    st.stop()

api_ok, api_message = backend_health()
revenue, revenue_year = latest_value(income_df, "Total Revenue")
operating_income, _ = latest_value(income_df, "Operating Income")
net_income, _ = latest_value(income_df, "Net Income")
free_cash_flow, _ = latest_value(cash_df, "Free Cash Flow")
latest_fy = "FY" + revenue_year[:4] if revenue_year else "Latest available FY"


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="eyebrow">Research workspace</div>'
        '<div style="font-size:1.2rem;font-weight:800;color:#eef5fa;letter-spacing:-.04em">TCS Intelligence</div>'
        '<div style="color:#8198ac;font-size:.75rem;margin-top:.25rem">Company research · India</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    if api_ok:
        st.markdown('<span class="status-pill"><span class="status-dot"></span> BACKEND REACHABLE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill" style="border-color:rgba(241,200,121,.35);color:#f1c879"><span class="status-dot" style="background:#f1c879"></span> BACKEND NOT REACHABLE</span>', unsafe_allow_html=True)
        st.caption(api_message)
        if st.button("Retry backend connection", key="retry_backend_health", use_container_width=True):
            backend_health.clear()
            st.rerun()
    st.caption("Financial data period: " + latest_fy)
    st.divider()
    st.markdown("**NAVIGATION**")
    navigation_options = [
        "Overview", "AI Research", "Financial performance",
        "Balance sheet", "Cash flow", "Markets & News", "Market Sentiment", "How it works",
    ]
    st.session_state.setdefault("active_dashboard_section", "Overview")
    for section in navigation_options:
        is_active = st.session_state.active_dashboard_section == section
        st.button(
            ("●  " if is_active else "") + section,
            key="nav_" + section.lower().replace(" ", "_").replace("&", "and"),
            type="primary" if is_active else "secondary",
            use_container_width=True,
            on_click=set_dashboard_section,
            args=(section,),
        )
    st.divider()
    st.markdown("**Company snapshot**")
    st.metric("Revenue · " + latest_fy, crore(revenue))
    st.metric("Net income · " + latest_fy, crore(net_income))
    st.metric("Free cash flow · " + latest_fy, crore(free_cash_flow))
    st.divider()
    st.markdown("**Explore with a question**")
    sidebar_prompts = [
        "Summarise TCS financial performance over the last five fiscal years.",
        "What are the main risks discussed in recent TCS disclosures?",
        "What has management said about generative AI and its revenue opportunity?",
        "Explain the trend in TCS operating cash flow and free cash flow.",
    ]
    for prompt in sidebar_prompts:
        st.button(prompt, key="sidebar_" + str(sidebar_prompts.index(prompt)), use_container_width=True, on_click=set_prompt, args=(prompt,))
    st.divider()
    st.caption("Answers are generated from retrieved company documents. Verify material figures against the cited source.")


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
header_left, header_right = st.columns([5, 1.35], vertical_alignment="center")
with header_left:
    st.markdown('<div class="eyebrow">Equity research · TATA CONSULTANCY SERVICES</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Financial intelligence,<br>with an evidence trail.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Explore company performance, interrogate disclosures and trace AI-generated research back to the source documents.</div>'
        '<div class="hero-meta">HYBRID RETRIEVAL&nbsp;&nbsp; / &nbsp;&nbsp;KNOWLEDGE GRAPH&nbsp;&nbsp; / &nbsp;&nbsp;GROUNDED GENERATION</div>',
        unsafe_allow_html=True,
    )
with header_right:
    if api_ok:
        st.markdown('<div style="text-align:right"><span class="status-pill"><span class="status-dot"></span> BACKEND REACHABLE</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right"><span class="status-pill" style="color:#f1c879;border-color:rgba(241,200,121,.35)">● BACKEND NOT CONFIRMED</span></div>', unsafe_allow_html=True)

st.write("")


# -------------------------------------------------------------------
# Main navigation
# -------------------------------------------------------------------
selected_section = st.session_state.active_dashboard_section

if selected_section == "Overview":
    section_heading("01 / Company dashboard", "TCS at a glance", f"Latest available annual financial data · {latest_fy}")
    cards = st.columns(4, gap="medium")
    with cards[0]:
        metric_card("Total revenue", crore(revenue), growth_text(income_df, "Total Revenue"))
    with cards[1]:
        metric_card("Operating income", crore(operating_income), growth_text(income_df, "Operating Income"))
    with cards[2]:
        metric_card("Net income", crore(net_income), growth_text(income_df, "Net Income"))
    with cards[3]:
        metric_card("Free cash flow", crore(free_cash_flow), growth_text(cash_df, "Free Cash Flow"))

    st.write("")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown('<div class="panel-title">Revenue trajectory</div><div class="panel-copy">Annual revenue, converted from reported rupees to ₹ crore.</div>', unsafe_allow_html=True)
        revenue_chart = chart_frame(income_df, ["Total Revenue"])
        if not revenue_chart.empty:
            fig = px.area(revenue_chart, x="Financial year", y="₹ Crore", markers=True)
            fig.update_traces(line=dict(color="#53d6df", width=3), fillcolor="rgba(83,214,223,.12)", marker=dict(size=7))
            st.plotly_chart(style_chart(fig, 350), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Revenue history is not available in the financial dataset.")
    with right:
        st.markdown('<div class="panel-title">Profit conversion</div><div class="panel-copy">Net income as a share of reported revenue.</div>', unsafe_allow_html=True)
        rev_s = metric_series(income_df, "Total Revenue")
        ni_s = metric_series(income_df, "Net Income")
        common = sorted(set(map(str, rev_s.index)) & set(map(str, ni_s.index)))
        margin_rows = []
        for year in common:
            if rev_s.get(year) and pd.notna(ni_s.get(year)):
                margin_rows.append({"Financial year": "FY" + year[:4], "Net margin": float(ni_s[year] / rev_s[year] * 100)})
        if margin_rows:
            margin_df = pd.DataFrame(margin_rows)
            fig = px.bar(margin_df, x="Financial year", y="Net margin", text=margin_df["Net margin"].map(lambda x: f"{x:.1f}%"))
            fig.update_traces(marker_color="#9be4c1", textposition="outside")
            fig.update_yaxes(title="Net margin (%)", ticksuffix="%", rangemode="tozero")
            st.plotly_chart(style_chart(fig, 350), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Profit margin history is not available.")

    st.markdown('<div class="panel"><div class="panel-title">Research desk</div><div class="panel-copy">Move from reported numbers to the underlying explanation. Ask about business segments, strategy, risks, AI initiatives or the drivers behind a financial trend.</div></div>', unsafe_allow_html=True)
    prompt_cols = st.columns(3)
    quick_prompts = [
        "What drove TCS revenue growth in the latest fiscal year?",
        "How has free cash flow changed over time?",
        "What are the key strategic risks for TCS?",
    ]
    for idx, prompt in enumerate(quick_prompts):
        with prompt_cols[idx]:
            st.button(prompt, key=f"overview_prompt_{idx}", use_container_width=True, on_click=set_prompt, args=(prompt,))


elif selected_section == "AI Research":
    section_heading("02 / Ask the documents", "AI research assistant", "Ask a focused question. Answers include the retrieved evidence returned by the research API.")
    if not api_ok:
        st.warning("The backend health check did not complete. This does not necessarily mean Groq or the whole service is offline; financial dashboards remain available. Retry your question in a moment.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""

    action_row = st.columns([5, 1])
    with action_row[0]:
        st.caption("Suggested starting points")
    with action_row[1]:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    suggestion_cols = st.columns(3)
    research_prompts = [
        "What does management say about AI and its impact on TCS?",
        "Summarise the key risks and uncertainties in TCS disclosures.",
        "Compare revenue and profitability trends across recent years.",
    ]
    for idx, prompt in enumerate(research_prompts):
        with suggestion_cols[idx]:
            st.button(prompt, key=f"research_prompt_{idx}", use_container_width=True, on_click=set_prompt, args=(prompt,))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    typed_question = st.chat_input("Ask about TCS strategy, financials, risks or disclosures…")
    question = typed_question or st.session_state.get("pending_prompt", "")
    if question:
        st.session_state.pending_prompt = ""
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching disclosures, ranking evidence and preparing a grounded response…"):
                try:
                    result = ask_question(question)
                    answer = result.get("answer") or result.get("detail") or "The API returned no answer."
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    render_sources(sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except requests.RequestException as exc:
                    message = f"The research service could not complete this request: {exc}"
                    st.error(message)
                    st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})
                except Exception as exc:
                    message = f"Unable to process the response: {exc}"
                    st.error(message)
                    st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})


elif selected_section == "Financial performance":
    section_heading("03 / Income statement", "Growth and profitability", "Reported annual financial metrics. Values are displayed in ₹ crore.")
    rev_value, _ = latest_value(income_df, "Total Revenue")
    gross_profit, _ = latest_value(income_df, "Gross Profit")
    op_income, _ = latest_value(income_df, "Operating Income")
    ni_value, _ = latest_value(income_df, "Net Income")
    metric_cols = st.columns(4)
    for col, label, value, foot in [
        (metric_cols[0], "Revenue", rev_value, growth_text(income_df, "Total Revenue")),
        (metric_cols[1], "Gross profit", gross_profit, growth_text(income_df, "Gross Profit")),
        (metric_cols[2], "Operating income", op_income, growth_text(income_df, "Operating Income")),
        (metric_cols[3], "Net income", ni_value, growth_text(income_df, "Net Income")),
    ]:
        with col:
            metric_card(label, crore(value), foot)

    st.write("")
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        df = chart_frame(income_df, ["Total Revenue", "Gross Profit"])
        if not df.empty:
            fig = px.line(df, x="Financial year", y="₹ Crore", color="Metric", markers=True, color_discrete_sequence=["#53d6df", "#9be4c1"])
            st.plotly_chart(style_chart(fig, 360), use_container_width=True, config={"displayModeBar": False})
    with chart_right:
        df = chart_frame(income_df, ["Operating Income", "Net Income"])
        if not df.empty:
            fig = px.bar(df, x="Financial year", y="₹ Crore", color="Metric", barmode="group", color_discrete_sequence=["#6fa9d8", "#e8bd73"])
            st.plotly_chart(style_chart(fig, 360), use_container_width=True, config={"displayModeBar": False})

    with st.expander("View reported income statement metrics"):
        visible = income_df.copy()
        visible.index.name = "Metric"
        st.dataframe((visible / 1e7).round(2), use_container_width=True, height=430)
        st.caption("Displayed values are converted from rupees to ₹ crore. Missing source values remain blank.")


elif selected_section == "Balance sheet":
    section_heading("04 / Financial position", "Balance sheet strength", "Assets, liabilities, equity and debt across the available fiscal years.")
    assets, _ = latest_value(balance_df, "Total Assets")
    liabilities, _ = latest_value(balance_df, "Total Liabilities Net Minority Interest")
    equity, _ = latest_value(balance_df, "Stockholders Equity")
    debt, _ = latest_value(balance_df, "Total Debt")
    metric_cols = st.columns(4)
    for col, label, value in [
        (metric_cols[0], "Total assets", assets),
        (metric_cols[1], "Total liabilities", liabilities),
        (metric_cols[2], "Shareholders' equity", equity),
        (metric_cols[3], "Total debt", debt),
    ]:
        with col:
            metric_card(label, crore(value), "Latest available annual figure")

    st.write("")
    balance_chart = chart_frame(balance_df, ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity"])
    if not balance_chart.empty:
        fig = px.area(balance_chart, x="Financial year", y="₹ Crore", color="Metric", markers=True, color_discrete_sequence=["#53d6df", "#e8bd73", "#9be4c1"])
        st.plotly_chart(style_chart(fig, 390), use_container_width=True, config={"displayModeBar": False})

    with st.expander("View reported balance sheet metrics"):
        visible = balance_df.copy()
        visible.index.name = "Metric"
        st.dataframe((visible / 1e7).round(2), use_container_width=True, height=430)


elif selected_section == "Cash flow":
    section_heading("05 / Cash generation", "Cash flow quality", "Operating cash flow, reinvestment and free cash flow.")
    operating_cash, _ = latest_value(cash_df, "Operating Cash Flow")
    capex, _ = latest_value(cash_df, "Capital Expenditure")
    end_cash, _ = latest_value(cash_df, "End Cash Position")
    metric_cols = st.columns(4)
    for col, label, value, foot in [
        (metric_cols[0], "Operating cash flow", operating_cash, growth_text(cash_df, "Operating Cash Flow")),
        (metric_cols[1], "Free cash flow", free_cash_flow, growth_text(cash_df, "Free Cash Flow")),
        (metric_cols[2], "Capital expenditure", abs(capex) if capex is not None else None, "Absolute investment outflow"),
        (metric_cols[3], "Cash position", end_cash, "Year-end cash position"),
    ]:
        with col:
            metric_card(label, crore(value), foot)

    st.write("")
    cash_chart = chart_frame(cash_df, ["Operating Cash Flow", "Free Cash Flow"])
    if not cash_chart.empty:
        fig = px.line(cash_chart, x="Financial year", y="₹ Crore", color="Metric", markers=True, color_discrete_sequence=["#53d6df", "#9be4c1"])
        st.plotly_chart(style_chart(fig, 370), use_container_width=True, config={"displayModeBar": False})

    with st.expander("View reported cash flow metrics"):
        visible = cash_df.copy()
        visible.index.name = "Metric"
        st.dataframe((visible / 1e7).round(2), use_container_width=True, height=430)


elif selected_section == "Markets & News":
    section_heading(
        "06 / Live public sources",
        "Markets & financial news",
        "Market data is retrieved from Yahoo Finance through yfinance; headlines are retrieved from Google News RSS. No paid data API key is required.",
    )
    market_col, period_col = st.columns([2, 1])
    with market_col:
        ticker = st.selectbox(
            "Instrument",
            ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "^NSEI", "^BSESN", "AAPL", "MSFT"],
            format_func=lambda value: {
                "TCS.NS": "TCS · NSE",
                "INFY.NS": "Infosys · NSE",
                "RELIANCE.NS": "Reliance Industries · NSE",
                "HDFCBANK.NS": "HDFC Bank · NSE",
                "ICICIBANK.NS": "ICICI Bank · NSE",
                "^NSEI": "NIFTY 50",
                "^BSESN": "SENSEX",
                "AAPL": "Apple · NASDAQ",
                "MSFT": "Microsoft · NASDAQ",
            }.get(value, value),
            key="market_ticker",
        )
    with period_col:
        history_period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="market_period")

    if st.button("↻ Refresh", key="refresh_market", use_container_width=True):
        load_market_quote.clear()
        load_market_history.clear()
        load_financial_news.clear()
        st.rerun()
    st.caption("Quotes are not guaranteed real-time. Provider availability and rate limits may vary.")

    try:
        with st.spinner("Loading public market data…"):
            quote = load_market_quote(ticker)
            history = load_market_history(ticker, history_period)
        price = float(quote["price"])
        change = quote.get("change")
        change_pct = quote.get("change_percent")
        currency = quote.get("currency") or "quote currency"
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Latest available close", f"{price:,.2f} {currency}", f"{change:+,.2f}" if change is not None else None)
        q2.metric("Previous close", f"{quote['previous_close']:,.2f}" if quote.get("previous_close") is not None else "—")
        q3.metric("Day high", f"{quote['day_high']:,.2f}")
        q4.metric("Day low", f"{quote['day_low']:,.2f}")
        if change_pct is not None:
            st.caption(f"Change vs previous available close: {change_pct:+.2f}% · Volume: {quote.get('volume', 0):,}")
        points = history.get("points", [])
        if points:
            price_frame = pd.DataFrame(points)
            price_frame["date"] = pd.to_datetime(price_frame["date"])
            fig = px.line(price_frame, x="date", y="close", title=f"{ticker} · daily closing prices")
            fig.update_traces(line=dict(color="#53d6df", width=2))
            st.plotly_chart(style_chart(fig, 360), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Source: {quote.get('source')} · Quote timestamp: {quote.get('market_data_timestamp', 'not supplied')}")
        st.caption(quote.get("price_note", "Market data may be delayed."))
    except requests.RequestException as exc:
        st.warning(f"Market data is temporarily unavailable: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        st.warning(f"Market data response could not be displayed: {exc}")

    st.divider()
    section_heading("06B / Headlines", "Financial news", "Search recent headlines by company, ticker or market topic.")
    news_query = st.text_input("News search", value="Tata Consultancy Services TCS", key="financial_news_query")
    if st.button("Search news", key="search_financial_news"):
        load_financial_news.clear()
    if news_query.strip():
        try:
            with st.spinner("Fetching recent headlines…"):
                news_payload = load_financial_news(news_query.strip())
            articles = news_payload.get("items", [])
            if not articles:
                st.info("No recent headlines were returned for this search.")
            for article in articles:
                title = str(article.get("title", "Untitled"))
                url = str(article.get("url", ""))
                publisher = str(article.get("publisher", "Publisher not identified"))
                published = str(article.get("published", ""))
                if url.startswith("https://"):
                    st.markdown(f"**[{title}]({url})**")
                else:
                    st.markdown(f"**{title}**")
                st.caption(" · ".join(part for part in [publisher, published] if part))
                st.divider()
            st.caption(f"Source: {news_payload.get('source', 'RSS feed')} · Retrieved: {news_payload.get('retrieved_at', 'timestamp unavailable')}")
        except requests.RequestException as exc:
            st.warning(f"News feed is temporarily unavailable: {exc}")



elif selected_section == "Market Sentiment":
    section_heading(
        "07 / NLP market intelligence",
        "Daily market-news sentiment",
        "Financial headlines scored with ProsusAI FinBERT. Sentiment describes the tone of coverage, not expected stock returns.",
    )
    sentiment_options = {
        "TCS · Tata Consultancy Services": "Tata Consultancy Services TCS",
        "Infosys": "Infosys India",
        "Reliance Industries": "Reliance Industries",
        "HDFC Bank": "HDFC Bank",
        "ICICI Bank": "ICICI Bank",
        "NIFTY 50 · Indian markets": "NIFTY 50 Indian stock market",
    }
    sentiment_company = st.selectbox("Company or market", list(sentiment_options), key="sentiment_company")
    sentiment_query = sentiment_options[sentiment_company]
    if st.button("↻ Refresh analysis", key="refresh_sentiment", use_container_width=True):
        load_daily_sentiment.clear()
        st.rerun()
    st.caption("Results are cached for 24 hours to reduce duplicate inference calls. Refresh manually to request a new analysis.")

    try:
        with st.spinner("Collecting headlines and running financial sentiment classification…"):
            sentiment_data = load_daily_sentiment(sentiment_query)
        counts = sentiment_data.get("counts", {})
        score = float(sentiment_data.get("sentiment_score", 0.0))
        label = str(sentiment_data.get("sentiment_label", "neutral")).title()
        st.markdown(
            f'<div class="panel"><div class="panel-copy">Overall headline tone</div>'
            f'<div style="font-size:2rem;font-weight:800;color:#9ce8e7">{label}</div>'
            f'<div class="panel-copy">Average sentiment score: {score:+.3f} · Range: -1 to +1</div></div>',
            unsafe_allow_html=True,
        )
        positive = int(counts.get("positive", 0))
        negative = int(counts.get("negative", 0))
        neutral = int(counts.get("neutral", 0))
        total = int(sentiment_data.get("article_count", positive + negative + neutral))
        sentiment_cols = st.columns(4)
        sentiment_cols[0].metric("Articles scored", total)
        sentiment_cols[1].metric("Positive", positive, f"{positive / total:.0%}" if total else None)
        sentiment_cols[2].metric("Negative", negative, f"{negative / total:.0%}" if total else None)
        sentiment_cols[3].metric("Neutral", neutral, f"{neutral / total:.0%}" if total else None)

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.markdown("**Headline distribution**")
            distribution = pd.DataFrame([
                {"Sentiment": "Positive", "Articles": positive},
                {"Sentiment": "Neutral", "Articles": neutral},
                {"Sentiment": "Negative", "Articles": negative},
            ])
            fig = px.pie(
                distribution, names="Sentiment", values="Articles", hole=.62,
                color="Sentiment",
                color_discrete_map={"Positive": "#75d6ad", "Neutral": "#8fa8bd", "Negative": "#e58b87"},
            )
            st.plotly_chart(style_chart(fig, 300), use_container_width=True, config={"displayModeBar": False})
        with chart_right:
            st.markdown("**Sentiment by publication date**")
            trend = pd.DataFrame(sentiment_data.get("trend", []))
            if not trend.empty:
                trend["date"] = pd.to_datetime(trend["date"])
                fig = px.bar(trend, x="date", y="sentiment_score", hover_data=["article_count"])
                fig.update_traces(marker_color="#53d6df")
                fig.add_hline(y=0, line_dash="dash", line_color="#91a3b5")
                fig.update_yaxes(title="Average sentiment", range=[-1, 1])
                st.plotly_chart(style_chart(fig, 300), use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Publication dates were not available for a daily trend.")

        st.markdown("**Classified headlines**")
        for article in sentiment_data.get("articles", []):
            headline = str(article.get("title", "Untitled"))
            article_url = str(article.get("url", ""))
            article_label = str(article.get("label", "neutral")).title()
            confidence = float(article.get("confidence", 0))
            if article_url.startswith("https://"):
                st.markdown(f"**[{headline}]({article_url})**")
            else:
                st.markdown(f"**{headline}**")
            st.caption(
                f"{article_label} · confidence {confidence:.0%} · "
                f"score {float(article.get('sentiment_score', 0)):+.2f} · "
                f"{article.get('publisher', 'Publisher not identified')} · "
                f"{article.get('published', 'Date unavailable')}"
            )
            st.divider()
        st.caption(
            f"Model: {sentiment_data.get('model')} · Source: {sentiment_data.get('source')} · "
            f"Retrieved: {sentiment_data.get('retrieved_at')}"
        )
        st.info(sentiment_data.get("methodology_note", "Headline sentiment is not investment advice."))
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        detail = ""
        if exc.response is not None:
            try:
                detail = str(exc.response.json().get("detail", ""))
            except ValueError:
                detail = ""
        if status == 503:
            st.warning(detail or "FinBERT is not configured. Add HF_TOKEN to the Render backend environment.")
        else:
            st.warning(f"Sentiment service returned HTTP {status}. {detail}")
    except requests.RequestException as exc:
        st.warning(f"Sentiment service is temporarily unavailable: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        st.warning(f"Sentiment response could not be displayed: {exc}")


elif selected_section == "How it works":
    section_heading("06 / System design", "How the research engine works", "A transparent overview of the retrieval and generation pipeline.")
    stages = [
        ("01", "Source documents", "TCS annual reports, quarterly disclosures, financial statements, earnings-call transcripts and company news."),
        ("02", "Evidence indexing", "Documents are chunked with source metadata; dense embeddings are stored in ChromaDB and lexical terms are indexed with BM25."),
        ("03", "Knowledge graph", "A NetworkX graph links financial entities and reporting periods to the evidence chunks where they appear."),
        ("04", "Hybrid retrieval", "Dense, lexical and graph evidence are combined using reciprocal-rank fusion before the LLM receives context."),
        ("05", "Grounded response", "The model is instructed to answer from retrieved evidence and return source references for verification."),
    ]
    for number, title, copy in stages:
        st.markdown(
            f'<div class="panel"><div style="display:flex;gap:1rem;align-items:flex-start">'
            f'<div style="font-family:DM Mono,monospace;color:#53d6df;font-size:.85rem;padding-top:.12rem">{number}</div>'
            f'<div><div class="panel-title">{title}</div><div class="panel-copy">{copy}</div></div></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("**Important scope notes**")
    st.info(
        "The current graph uses rule-based financial entity extraction and evidence co-mention edges. "
        "Those edges do not imply causality or directional business relationships. The graph is a local "
        "NetworkX index, not a Neo4j deployment. Retrieval metrics should only be reported after a manually "
        "reviewed benchmark has been completed."
    )
    st.caption("Research support only — not investment advice. Confirm material statements against original filings.")


st.markdown(
    '<div class="footer">TCS FINANCIAL INTELLIGENCE &nbsp;·&nbsp; HYBRID GRAPH RAG &nbsp;·&nbsp; BUILT WITH STREAMLIT, FASTAPI, CHROMADB AND GROQ</div>',
    unsafe_allow_html=True,
)

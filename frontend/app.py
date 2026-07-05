
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

from utils import (
    load_financials,
    get_metric
)

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="TCS Financial Intelligence Agent",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Backend Health Check
# -----------------------------

API_URL = "https://financial-intelligence-agent-pmgy.onrender.com/ask"
HEALTH_URL = "https://financial-intelligence-agent-pmgy.onrender.com/"


def ask_question(question):

    response = requests.get(
        API_URL,
        params={
            "question": question
        },
        timeout=180
    )

    if response.status_code != 200:
        raise Exception(
            f"API Error: {response.status_code}"
        )

    return response.json()


try:

    health = requests.get(
        HEALTH_URL,
        timeout=10
    )

    if health.status_code != 200:

        st.error("Backend unavailable")
        st.stop()

except Exception:

    st.error("Cannot connect to backend")
    st.stop()

# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Styling
# -----------------------------

st.markdown(
    """
    <style>
    .big-font {
        font-size:28px !important;
        font-weight:bold;
        color:#1E3A8A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<p class="big-font">📊 TCS Financial Intelligence Agent</p>',
    unsafe_allow_html=True
)

st.caption(
    "AI-Powered Equity Research Assistant"
)

# -----------------------------
# Load Financials
# -----------------------------

@st.cache_data
def load_cached_financials():
    return load_financials()

income_df, balance_df, cashflow_df = (
    load_cached_financials()
)
# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📈 TCS Snapshot")

st.sidebar.metric(
    "Revenue FY2024",
    "₹240,893 Cr"
)

st.sidebar.metric(
    "Growth",
    "6.8%"
)

st.sidebar.metric(
    "Employees",
    "600K+"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Suggested Questions"
)

suggestions = [
    "What was TCS revenue in FY2024?",
    "Summarize FY2024 performance",
    "What did TCS say about AI?",
    "What are key risks?"
]

selected_question = None

for q in suggestions:

    if st.sidebar.button(q):
        selected_question = q

# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💬 Chat",
        "📈 Financials",
        "🏦 Balance Sheet",
        "💰 Cash Flow",
        "🧠 Executive Insights"
    ]
)

# ==================================================
# CHAT TAB
# ==================================================

with tab1:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask a question about TCS..."
    )

    if selected_question:
        question = selected_question

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner(
                        "🔍 Searching reports and generating answer..."
        ):

            try:

                data = ask_question(question)

            except Exception as e:

                st.error(str(e))
                st.stop()

        with st.chat_message("assistant"):

            if "answer" in data:

                answer = data["answer"]

                st.success(
                    "Answer Generated"
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            else:

                st.error(
                    data.get(
                        "error",
                        "Unknown API Error"
                    )
                )

                st.stop()

            sources = data.get(
                "sources",
                []
            )

            if sources:

                with st.expander(
                    f"📚 Sources ({len(sources)})"
                ):
                    


                    import os

                    for source in sources:

                        file_name = os.path.basename(source)

                        st.markdown(
                                f"- {file_name}"
                                     )

# ==================================================
# FINANCIALS TAB
# ==================================================

with tab2:

    st.subheader(
        "Financial Performance"
    )

    revenue = get_metric(
        income_df,
        "Total Revenue"
    )

    net_income = get_metric(
        income_df,
        "Net Income"
    )

    fcf = get_metric(
        cashflow_df,
        "Free Cash Flow"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Revenue FY2024",
            f"₹{revenue['2024-03-31']/1e7:.0f} Cr"
        )

    with col2:
        st.metric(
            "Net Income FY2024",
            f"₹{net_income['2024-03-31']/1e7:.0f} Cr"
        )

    with col3:
        st.metric(
            "Free Cash Flow FY2024",
            f"₹{fcf['2024-03-31']/1e7:.0f} Cr"
        )

    years = [
        "2022-03-31",
        "2023-03-31",
        "2024-03-31",
        "2025-03-31",
        "2026-03-31"
    ]

    revenue_df = pd.DataFrame(
        {
            "Year": years,
            "Revenue": [
                revenue[y]
                for y in years
            ]
        }
    )

    fig = px.line(
        revenue_df,
        x="Year",
        y="Revenue",
        markers=True,
        title="Revenue Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    profit_df = pd.DataFrame(
        {
            "Year": years,
            "Profit": [
                net_income[y]
                for y in years
            ]
        }
    )

    fig2 = px.bar(
        profit_df,
        x="Year",
        y="Profit",
        title="Net Income Trend"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==================================================
# BALANCE SHEET TAB
# ==================================================

with tab3:

    st.subheader(
        "Balance Sheet Snapshot"
    )

    total_assets = get_metric(
        balance_df,
        "Total Assets"
    )

    total_debt = get_metric(
        balance_df,
        "Total Debt"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Assets FY2024",
            f"₹{total_assets['2024-03-31']/1e7:.0f} Cr"
        )

    with col2:

        st.metric(
            "Total Debt FY2024",
            f"₹{total_debt['2024-03-31']/1e7:.0f} Cr"
        )

# ==================================================
# CASH FLOW TAB
# ==================================================

with tab4:

    st.subheader(
        "Cash Flow Analysis"
    )

    free_cash_flow = get_metric(
        cashflow_df,
        "Free Cash Flow"
    )

    capex = get_metric(
        cashflow_df,
        "Capital Expenditure"
    )

    cash_position = get_metric(
        cashflow_df,
        "End Cash Position"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Free Cash Flow",
            f"₹{free_cash_flow['2024-03-31']/1e7:.0f} Cr"
        )

    with col2:

        st.metric(
            "Capital Expenditure",
            f"₹{capex['2024-03-31']/1e7:.0f} Cr"
        )

    with col3:

        st.metric(
            "Cash Position",
            f"₹{cash_position['2024-03-31']/1e7:.0f} Cr"
        )

# ==================================================
# EXECUTIVE INSIGHTS
# ==================================================

with tab5:

    st.subheader(
        "AI Executive Insights"
    )

    if st.button(
        "Generate Executive Insights"
    ):

        questions = [
            "What are the key growth drivers for TCS?",
            "What are the major risks facing TCS?",
            "What did management say about AI?"
        ]

        with st.spinner(
            "Generating executive insights..."
        ):

            for q in questions:

                try:
                    data = ask_question(q)

                except Exception as e:

                    st.error(str(e))
                    continue

                   

                    st.markdown(
                        f"### {q}"
                    )

                    if "answer" in data:

                        st.write(
                            data["answer"]
                        )

                    else:

                        st.error(
                            data.get(
                                "error",
                                "Unknown API Error"
                            )
                        )

                    st.divider()

                except Exception as e:

                    st.error(
                        f"Failed to fetch insight: {e}"
                    )
                    
                    # -----------------------------
# Footer
# -----------------------------

                    st.divider()

                    st.caption(
                            "Built with FastAPI • ChromaDB • BAAI/bge-small-en-v1.5 • Groq LLM • Streamlit"
                            )

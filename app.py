"""
Crypto Agent - on-demand web app.
Open it any time: live market overview, top risk-adjusted picks,
look up any coin, and ask questions in plain English.

Analysis only. Not financial advice. Not a price predictor.
"""
import json
import os
import re
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

import analyze as A

DEMO = os.environ.get("CRYPTO_DEMO") == "1"
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

st.set_page_config(page_title="Crypto Agent", layout="wide")

DISCLAIMER = ("Educational analysis only, not financial advice. Crypto can lose most of its value, "
              "and nobody can reliably predict prices. Only use money you can afford to lose.")


# ----------------------------- DATA HELPERS -----------------------------
def load_snapshot():
    try:
        df = pd.read_csv(A.SNAPSHOT_FILE)
        with open(A.META_FILE, encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta
    except Exception:
        return None, {}


@st.cache_data(ttl=300, show_spinner=False)
def market_overview():
    if DEMO:
        return {"mcap": 3.1e12, "mcap_change": 1.2, "btc_dom": 54.3, "volume": 9.8e10}
    data = A.get_json(f"{A.BASE}/global", tries=2)
    try:
        d = data["data"]
        return {"mcap": d["total_market_cap"]["usd"],
                "mcap_change": d["market_cap_change_percentage_24h_usd"],
                "btc_dom": d["market_cap_percentage"]["btc"],
                "volume": d["total_volume"]["usd"]}
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fear_greed():
    if DEMO:
        return 55, "Greed"
    return A.fetch_fear_greed()


def run_full_refresh(n=20):
    """Live analysis of the top n coins, with a progress bar."""
    A.TOP_N = n
    bar = st.progress(0.0, text="Fetching coin list...")
    coins = A.fetch_top_coins()
    hist = {}
    for i, c in enumerate(coins, 1):
        bar.progress(i / len(coins), text=f"Analysing {c['name']} ({i}/{len(coins)})")
        h = A.fetch_history(c["id"])
        if h is not None:
            hist[c["id"]] = h
        time.sleep(A.SLEEP_SECONDS)
    bar.empty()
    if "bitcoin" not in hist:
        raise RuntimeError("Could not get Bitcoin data (probably rate-limited). Try again in a minute.")
    btc_ret = hist["bitcoin"].pct_change().dropna()
    rows = [A.analyse_coin(c, hist[c["id"]], btc_ret) for c in coins if c["id"] in hist]
    rows = [r for r in rows if r]
    return A.score_and_rank(pd.DataFrame(rows))


def lookup_coin(coin_id):
    """Live analysis of one coin (any coin on CoinGecko)."""
    if DEMO:
        coins, hist, _ = A.demo_data()
        coin = next(c for c in coins if c["id"] == coin_id)
        prices, btc = hist[coin_id], hist["bitcoin"]
    else:
        info = A.get_json(f"{A.BASE}/coins/markets", {"vs_currency": "usd", "ids": coin_id}, tries=3)
        if not info:
            raise RuntimeError("Coin data not available right now.")
        coin = info[0]
        prices = A.fetch_history(coin_id)
        time.sleep(A.SLEEP_SECONDS if A.SLEEP_SECONDS < 4 else 4)
        btc = A.fetch_history("bitcoin")
        if prices is None or btc is None:
            raise RuntimeError("Price history not available (rate limit?). Wait a minute and try again.")
    row = A.analyse_coin(coin, prices, btc.pct_change().dropna())
    if row is None:
        raise RuntimeError("Not enough price history for this coin to analyse it.")
    row["risk"] = ("Lower" if row["ann_vol"] < 0.60 and row["max_dd"] > -0.45
                   else "High" if row["ann_vol"] > 1.0 or row["max_dd"] < -0.65 else "Medium")
    return row, prices


def search_coins(q):
    if DEMO:
        coins, _, _ = A.demo_data()
        return [(c["id"], f"{c['name']} ({c['symbol'].upper()})") for c in coins if q.lower() in c["name"].lower()]
    data = A.get_json(f"{A.BASE}/search", {"query": q}, tries=2)
    try:
        return [(c["id"], f"{c['name']} ({c['symbol'].upper()})") for c in data["coins"][:8]]
    except Exception:
        return []


# ----------------------------- FORMATTING -----------------------------
def show_table(df):
    t = pd.DataFrame({
        "Coin": df["symbol"] + " - " + df["name"],
        "Score": df["score"].round(0).astype(int),
        "Risk": df["risk"],
        "Trend": df["trend"],
        "Price": df["price"].map(A.price_fmt),
        "7d": df["roi_7d"].map(A.pct), "30d": df["roi_30d"].map(A.pct),
        "90d": df["roi_90d"].map(A.pct), "1y": df["roi_1y"].map(A.pct),
        "Volatility (yr)": df["ann_vol"].map(lambda x: A.pct(x, 0)),
        "Worst drop": df["max_dd"].map(lambda x: A.pct(x, 0)),
        "Sharpe": df["sharpe"].round(2),
        "Chance up 30d": df["p_up_30d"].map(lambda x: A.pct(x, 0)),
    })
    st.dataframe(t, hide_index=True, use_container_width=True)


# ----------------------------- AI CHAT -----------------------------
def build_context(df, meta, question):
    cols = ["symbol", "name", "price", "score", "risk", "trend", "roi_7d", "roi_30d", "roi_90d",
            "roi_1y", "ann_vol", "max_dd", "sharpe", "sortino", "beta_btc", "rsi", "p_up_30d",
            "range_lo", "range_hi"]
    ctx = df[cols].round(3).to_csv(index=False)
    fg = fear_greed()
    mo = market_overview()
    parts = [f"Data time (UTC): {meta.get('generated_utc', 'live session')}",
             f"Today: {dt.date.today()}",
             f"Fear & Greed: {fg[0]} ({fg[1]})"]
    if mo:
        parts.append(f"Total crypto market cap: ${mo['mcap']/1e12:.2f}T, 24h change {mo['mcap_change']:.1f}%, "
                     f"BTC dominance {mo['btc_dom']:.1f}%")
    parts.append("Coin metrics (roi = return, ann_vol = yearly volatility, max_dd = worst drop in a year, "
                 "p_up_30d = share of 5000 history-based simulations ending 30 days up):\n" + ctx)
    return "\n".join(parts)


SYSTEM_PROMPT = (
    "You are a careful crypto market analyst helping a beginner. Use ONLY the data provided; "
    "never invent prices, returns or news. If the answer is not in the data, say so and suggest the "
    "'Look up any coin' tab. When asked which coin to invest in for low risk and high return, name 2-3 "
    "candidates from the data with the numbers that justify them (risk-adjusted score, volatility, worst drop, "
    "Sharpe), say what could go wrong, and note that no coin is safe. Never promise or predict returns; "
    "describe the 30-day outlook as a history-based range, not a forecast. Keep answers under 200 words, "
    "plain language, and end with a one-line reminder that this is not financial advice."
)


def ask_gemini(history, context):
    contents = []
    for m in history[-8:]:
        contents.append({"role": "user" if m["role"] == "user" else "model",
                         "parts": [{"text": m["content"]}]})
    # Attach the data to the latest user message
    contents[-1]["parts"][0]["text"] = f"DATA:\n{context}\n\nQUESTION: {contents[-1]['parts'][0]['text']}"
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_KEY},
        json={"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# ----------------------------- PAGE -----------------------------
st.title("Crypto Agent")
st.caption(DISCLAIMER)

snap_df, meta = load_snapshot()
if "df" not in st.session_state and snap_df is not None:
    st.session_state["df"] = snap_df
    st.session_state["df_time"] = meta.get("generated_utc", "")

# --- Market overview (live) ---
mo = market_overview()
fg = fear_greed()
c1, c2, c3, c4 = st.columns(4)
if mo:
    c1.metric("Total crypto market cap", f"${mo['mcap'] / 1e12:.2f}T", f"{mo['mcap_change']:.1f}% (24h)")
    c2.metric("Bitcoin dominance", f"{mo['btc_dom']:.1f}%")
    c3.metric("24h trading volume", f"${mo['volume'] / 1e9:.0f}B")
else:
    c1.write("Market overview unavailable right now.")
if fg and fg[0] is not None:
    c4.metric("Fear & Greed", f"{fg[0]}/100", fg[1], delta_color="off")

tab_top, tab_coin, tab_chat = st.tabs(["Top picks", "Look up any coin", "Ask the agent"])

# --- Tab 1: ranking ---
with tab_top:
    df = st.session_state.get("df")
    if df is None:
        st.info("No saved analysis yet. Click the button below to run one (takes a few minutes).")
    else:
        when = st.session_state.get("df_time", "")
        st.caption(f"Analysis time (UTC): {when}" + ("  |  DEMO DATA" if meta.get("demo") or DEMO else ""))

    if st.button("Refresh full analysis now (2-4 minutes)"):
        try:
            st.session_state["df"] = run_full_refresh(20)
            st.session_state["df_time"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            st.rerun()
        except Exception as e:
            st.error(f"Refresh failed: {e}")

    df = st.session_state.get("df")
    if df is not None:
        risk_choice = st.radio("Show", ["All", "Lower risk only", "Medium or lower"], horizontal=True)
        view = df
        if risk_choice == "Lower risk only":
            view = df[df["risk"] == "Lower"]
        elif risk_choice == "Medium or lower":
            view = df[df["risk"].isin(["Lower", "Medium"])]
        if view.empty:
            st.warning("No coin fits that filter right now.")
        else:
            st.subheader("Best risk-adjusted scores")
            show_table(view.head(15))
            st.bar_chart(view.head(10).set_index("symbol")["score"])
        st.caption("Score ranks coins against each other on return per unit of risk, worst drop, size and "
                   "90-day return. It is not a buy signal.")

# --- Tab 2: single coin ---
with tab_coin:
    q = st.text_input("Coin name or symbol (for example: solana, sui, pepe)")
    if q:
        options = search_coins(q)
        if not options:
            st.warning("No coin found. Try another name.")
        else:
            labels = [o[1] for o in options]
            choice = st.selectbox("Which one?", labels)
            if st.button("Analyse this coin"):
                coin_id = options[labels.index(choice)][0]
                with st.spinner("Fetching live data..."):
                    try:
                        st.session_state["lookup"] = lookup_coin(coin_id)
                    except Exception as e:
                        st.session_state.pop("lookup", None)
                        st.error(str(e))
    if "lookup" in st.session_state:
        row, prices = st.session_state["lookup"]
        st.subheader(f"{row['name']} ({row['symbol']})")
        m = st.columns(4)
        m[0].metric("Price", A.price_fmt(row["price"]))
        m[1].metric("Market cap", A.usd(row["market_cap"]))
        m[2].metric("Risk level", row["risk"])
        m[3].metric("Trend", row["trend"])
        r = st.columns(4)
        r[0].metric("7 days", A.pct(row["roi_7d"]))
        r[1].metric("30 days", A.pct(row["roi_30d"]))
        r[2].metric("90 days", A.pct(row["roi_90d"]))
        r[3].metric("1 year", A.pct(row["roi_1y"]))
        k = st.columns(4)
        k[0].metric("Volatility (yearly)", A.pct(row["ann_vol"], 0))
        k[1].metric("Worst drop (1y)", A.pct(row["max_dd"], 0))
        k[2].metric("Sharpe ratio", f"{row['sharpe']:.2f}")
        k[3].metric("Beta vs Bitcoin", f"{row['beta_btc']:.2f}")
        st.line_chart(prices.rename("Price (USD)"))
        st.write(f"**30-day outlook from its own history (not a forecast):** about {A.pct(row['p_up_30d'], 0)} "
                 f"chance of ending up. Bad case (5%): {A.pct(row['range_lo'], 0)}. "
                 f"Good case (95%): {A.pct(row['range_hi'], 0)}. RSI is {row['rsi']:.0f}.")

# --- Tab 3: chat ---
with tab_chat:
    df = st.session_state.get("df")
    if df is None:
        st.info("Run the analysis first on the 'Top picks' tab, then ask questions here.")
    elif not GEMINI_KEY:
        st.warning("To chat in plain English, add a free Gemini key as a secret named GEMINI_API_KEY "
                   "(see the README). Meanwhile the other two tabs work without it.")
    else:
        st.caption("Try: 'Which coin has the best return for low risk?', 'Is SOL risky?', "
                   "'Compare BTC and ETH', 'What is the market mood?'")
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        for m in st.session_state["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        question = st.chat_input("Ask about coins, risk, returns or the market...")
        if question:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        ctx = build_context(df, meta, question)
                        answer = ask_gemini(st.session_state["messages"], ctx)
                    except Exception as e:
                        answer = f"Sorry, the AI service did not respond ({e}). Please try again."
                st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

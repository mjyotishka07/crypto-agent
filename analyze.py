"""
Crypto Analysis Agent
---------------------
Fetches free market data from CoinGecko, computes ROI and risk metrics,
ranks coins by risk-adjusted return, and writes REPORT.md.

This is an analysis tool, NOT financial advice and NOT a price predictor.

Usage:
    python analyze.py          # live data
    python analyze.py --demo   # fake data, for testing without internet
"""
import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

# ----------------------------- SETTINGS -----------------------------
TOP_N = 30            # how many top coins (by market cap) to analyse
DAYS = 365            # history length
SLEEP_SECONDS = 3 if os.environ.get("COINGECKO_API_KEY", "").strip() else 7  # pause between API calls (free tier rate limit)
REPORT_FILE = "REPORT.md"
SNAPSHOT_FILE = "ranking.csv"   # read by the web app so it opens instantly
META_FILE = "meta.json"
RISK_FREE = 0.0

# Stablecoins don't move, so they would look "low risk" and distort rankings.
STABLECOINS = {
    "tether", "usd-coin", "dai", "first-digital-usd", "ethena-usde", "usds",
    "binance-usd", "true-usd", "paypal-usd", "usdd", "frax", "stasis-eur",
    "usd1-wlfi", "ripple-usd", "global-dollar", "bridged-usdc-polygon-pos-bridge",
    "wrapped-bitcoin", "staked-ether", "wrapped-steth", "weth", "wrapped-eeth",
    "coinbase-wrapped-btc", "wrapped-beacon-eth", "binance-staked-sol",
}
# Wrapped / staked copies are excluded too, since they just duplicate BTC/ETH/SOL.

API_KEY = os.environ.get("COINGECKO_API_KEY", "").strip()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

BASE = "https://api.coingecko.com/api/v3"
HEADERS = {"accept": "application/json", "User-Agent": "crypto-agent/1.0"}
if API_KEY:
    HEADERS["x-cg-demo-api-key"] = API_KEY


# ----------------------------- DATA -----------------------------
def get_json(url, params=None, tries=5):
    """GET with retries, because free APIs sometimes say 'slow down'."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  request problem ({e}); retry {attempt + 1}/{tries}")
            time.sleep(10)
    return None


def fetch_top_coins():
    data = get_json(
        f"{BASE}/coins/markets",
        {"vs_currency": "usd", "order": "market_cap_desc",
         "per_page": TOP_N + 25, "page": 1, "sparkline": "false"},
    )
    if not data:
        raise RuntimeError("Could not fetch coin list from CoinGecko.")
    coins = [c for c in data if c["id"] not in STABLECOINS]
    return coins[:TOP_N]


def fetch_history(coin_id):
    data = get_json(
        f"{BASE}/coins/{coin_id}/market_chart",
        {"vs_currency": "usd", "days": DAYS, "interval": "daily"},
    )
    if not data or "prices" not in data:
        return None
    df = pd.DataFrame(data["prices"], columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    s = df.drop_duplicates("date", keep="last").set_index("date")["price"]
    return s.astype(float)


def fetch_fear_greed():
    data = get_json("https://api.alternative.me/fng/", {"limit": 1})
    try:
        item = data["data"][0]
        return int(item["value"]), item["value_classification"]
    except Exception:
        return None, None


def demo_data():
    """Fake data so the code can be tested offline."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=DAYS)
    names = ["bitcoin", "ethereum", "solana", "ripple", "cardano", "dogecoin",
             "chainlink", "avalanche-2", "polkadot", "litecoin"]
    coins, hist = [], {}
    for i, n in enumerate(names):
        drift = rng.normal(0.0004, 0.0006)
        vol = 0.02 + 0.006 * i
        r = rng.normal(drift, vol, DAYS)
        hist[n] = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        coins.append({"id": n, "symbol": n[:4], "name": n.title(),
                      "market_cap": 5e11 / (i + 1) ** 1.5,
                      "total_volume": 2e10 / (i + 1),
                      "current_price": float(hist[n].iloc[-1])})
    return coins, hist, (55, "Greed")


# ----------------------------- METRICS -----------------------------
def rsi(prices, n=14):
    delta = prices.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def roi(prices, days):
    if len(prices) <= days:
        return np.nan
    return float(prices.iloc[-1] / prices.iloc[-1 - days] - 1)


def max_drawdown(prices):
    peak = prices.cummax()
    return float((prices / peak - 1).min())


def bootstrap_outlook(returns, horizon=30, sims=5000, seed=1):
    """
    Not a prediction. Re-shuffles the coin's own past daily returns to show
    what range of 30-day outcomes its past behaviour would allow.
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    if len(r) < 60:
        return np.nan, np.nan, np.nan
    draws = rng.choice(r, size=(sims, horizon), replace=True)
    final = np.exp(np.log1p(draws).sum(axis=1)) - 1
    return float((final > 0).mean()), float(np.percentile(final, 5)), float(np.percentile(final, 95))


def analyse_coin(coin, prices, btc_returns):
    prices = prices.dropna()
    if len(prices) < 90:
        return None
    ret = prices.pct_change().dropna()
    vol = ret.std()
    ann_vol = vol * np.sqrt(365)
    sharpe = (ret.mean() - RISK_FREE / 365) / vol * np.sqrt(365) if vol > 0 else np.nan
    downside = ret[ret < 0].std()
    sortino = ret.mean() / downside * np.sqrt(365) if downside and downside > 0 else np.nan

    joined = pd.concat([ret, btc_returns], axis=1, join="inner").dropna()
    beta = np.nan
    if len(joined) > 30 and joined.iloc[:, 1].var() > 0:
        beta = float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var())

    sma50 = prices.rolling(50).mean().iloc[-1]
    sma200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else np.nan
    last = prices.iloc[-1]
    r = rsi(prices)

    # Simple trend reading
    pts = 0
    pts += 1 if last > sma50 else -1
    if not np.isnan(sma200):
        pts += 1 if last > sma200 else -1
    pts += 1 if roi(prices, 30) > 0 else -1
    if r > 75:
        pts -= 1
    elif r < 30:
        pts += 1
    trend = "Bullish" if pts >= 2 else "Bearish" if pts <= -2 else "Neutral"

    p_up, lo, hi = bootstrap_outlook(ret)

    return {
        "id": coin["id"], "symbol": coin["symbol"].upper(), "name": coin["name"],
        "price": float(coin.get("current_price") or last),
        "market_cap": float(coin.get("market_cap") or 0),
        "volume_24h": float(coin.get("total_volume") or 0),
        "roi_7d": roi(prices, 7), "roi_30d": roi(prices, 30),
        "roi_90d": roi(prices, 90), "roi_1y": roi(prices, 365 if len(prices) > 365 else len(prices) - 1),
        "ann_vol": float(ann_vol), "max_dd": max_drawdown(prices),
        "sharpe": float(sharpe), "sortino": float(sortino), "beta_btc": beta,
        "rsi": r, "trend": trend,
        "p_up_30d": p_up, "range_lo": lo, "range_hi": hi,
    }


def score_and_rank(df):
    """Higher score = better risk-adjusted return, lower drawdown, more liquid."""
    pr = lambda s, asc=True: s.rank(pct=True, ascending=asc)
    df["liq"] = np.log10(df["market_cap"].clip(lower=1))
    df["score"] = 100 * (
        0.30 * pr(df["sharpe"]) +
        0.15 * pr(df["sortino"]) +
        0.25 * pr(df["max_dd"]) +          # less negative drawdown = higher rank
        0.15 * pr(df["liq"]) +
        0.15 * pr(df["roi_90d"])
    )
    # Penalise downtrends and tiny liquidity
    df.loc[df["trend"] == "Bearish", "score"] -= 8
    df.loc[df["trend"] == "Bullish", "score"] += 4
    df["risk"] = np.where(
        (df["ann_vol"] < 0.60) & (df["max_dd"] > -0.45), "Lower",
        np.where((df["ann_vol"] > 1.0) | (df["max_dd"] < -0.65), "High", "Medium"))
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ----------------------------- REPORT -----------------------------
def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def usd(x):
    if pd.isna(x):
        return "n/a"
    if x >= 1e9:
        return f"${x / 1e9:.1f}B"
    if x >= 1e6:
        return f"${x / 1e6:.0f}M"
    return f"${x:,.0f}"


def price_fmt(x):
    return f"${x:,.2f}" if x >= 1 else f"${x:.4f}"


def llm_summary(df, fg):
    if not GEMINI_KEY:
        return None
    top = df.head(8)[["symbol", "score", "risk", "trend", "roi_30d", "roi_90d",
                      "ann_vol", "max_dd", "sharpe", "p_up_30d"]].round(3).to_string(index=False)
    prompt = (
        "You are a cautious crypto market analyst. Using ONLY the numbers below, "
        "write a short daily briefing (max 180 words) for a beginner: overall market mood, "
        "which 2-3 coins look best on risk-adjusted terms and why, which are risky, "
        "and one reminder that nothing is guaranteed. Do not invent numbers or prices. "
        f"Fear & Greed index: {fg[0]} ({fg[1]}).\n\n{top}"
    )
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            params={"key": GEMINI_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"AI summary skipped: {e}")
        return None


def build_report(df, fg, demo=False):
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    L = []
    L.append("# Daily Crypto Analysis Report")
    L.append(f"_Generated {now:%A, %d %B %Y, %H:%M} IST_" + ("  \n**DEMO DATA - not real**" if demo else ""))
    L.append("")
    L.append("> Educational analysis only, not financial advice. Crypto can lose most of its value. "
             "Nobody can reliably predict prices. Never invest money you can't afford to lose.")
    L.append("")

    if fg[0] is not None:
        L.append(f"**Market mood (Fear & Greed Index):** {fg[0]}/100, {fg[1]}")
        L.append("")

    summary = llm_summary(df, fg)
    if summary:
        L.append("## AI briefing")
        L.append(summary)
        L.append("")

    L.append("## Top 5 by risk-adjusted score")
    L.append("Ranked on Sharpe/Sortino ratio (return per unit of risk), size of worst drop, "
             "liquidity, and 90-day return. Stablecoins excluded.")
    L.append("")
    L.append("| # | Coin | Score | Risk | Trend | 30d | 90d | 1y | Worst drop | Sharpe |")
    L.append("|---|------|-------|------|-------|-----|-----|----|-----------|--------|")
    for i, r in df.head(5).iterrows():
        L.append(f"| {i + 1} | **{r.symbol}** ({r['name']}) | {r.score:.0f} | {r.risk} | {r.trend} | "
                 f"{pct(r.roi_30d)} | {pct(r.roi_90d)} | {pct(r.roi_1y)} | {pct(r.max_dd, 0)} | {r.sharpe:.2f} |")
    L.append("")

    lower = df[df.risk == "Lower"].head(5)
    L.append("## Lower-risk group")
    L.append(", ".join(f"{r.symbol}" for _, r in lower.iterrows()) if len(lower) else
             "No coin currently qualifies as lower risk. Everything is volatile right now.")
    L.append("")
    L.append("_Lower risk here is relative: annualised volatility under 60% and worst drop under 45%. "
             "That is still far riskier than stocks or bonds._")
    L.append("")

    L.append("## 30-day outlook (based on each coin's own past behaviour)")
    L.append("This is **not a forecast**. It re-shuffles past daily moves to show the range of outcomes "
             "that history would allow.")
    L.append("")
    L.append("| Coin | Chance up in 30d | Bad case (5%) | Good case (95%) |")
    L.append("|------|------------------|---------------|-----------------|")
    for _, r in df.head(10).iterrows():
        L.append(f"| {r.symbol} | {pct(r.p_up_30d, 0)} | {pct(r.range_lo, 0)} | {pct(r.range_hi, 0)} |")
    L.append("")

    L.append("## Full table")
    L.append("| Coin | Price | Mkt cap | 7d | 30d | 90d | 1y | Volatility (yr) | Worst drop | Sharpe | Beta vs BTC | RSI | Trend | Risk | Score |")
    L.append("|------|-------|---------|----|-----|-----|----|-----------------|-----------|--------|-------------|-----|-------|------|-------|")
    for _, r in df.iterrows():
        L.append(f"| {r.symbol} | {price_fmt(r.price)} | {usd(r.market_cap)} | {pct(r.roi_7d)} | {pct(r.roi_30d)} | "
                 f"{pct(r.roi_90d)} | {pct(r.roi_1y)} | {pct(r.ann_vol, 0)} | {pct(r.max_dd, 0)} | "
                 f"{r.sharpe:.2f} | {r.beta_btc:.2f} | {r.rsi:.0f} | {r.trend} | {r.risk} | {r.score:.0f} |")
    L.append("")

    L.append("## How to read this")
    L.append("- **ROI**: how much the price changed over that period.\n"
             "- **Volatility**: how wildly the price swings. Higher means riskier.\n"
             "- **Worst drop**: biggest fall from a peak in the past year.\n"
             "- **Sharpe / Sortino**: return per unit of risk. Above 1 is good, below 0 is poor.\n"
             "- **Beta vs BTC**: above 1 means it moves more than Bitcoin, below 1 less.\n"
             "- **RSI**: above 70 often means overheated, below 30 oversold.\n"
             "- **Trend**: simple reading from moving averages, 30-day return and RSI.\n"
             "- **Score**: relative ranking among these coins only, not a buy signal.")
    L.append("")
    L.append("Data: CoinGecko, alternative.me. Past performance does not predict future results.")
    return "\n".join(L)


def save_snapshot(df, fg, demo=False):
    """Save the ranking so the web app can open instantly with fresh-ish data."""
    import json
    df.to_csv(SNAPSHOT_FILE, index=False)
    meta = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fear_greed": fg[0], "fear_greed_label": fg[1], "demo": bool(demo),
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f)


# ----------------------------- MAIN -----------------------------
def main():
    demo = "--demo" in sys.argv
    if demo:
        coins, hist, fg = demo_data()
    else:
        print("Fetching coin list...")
        coins = fetch_top_coins()
        fg = fetch_fear_greed()
        hist = {}
        for i, c in enumerate(coins, 1):
            print(f"[{i}/{len(coins)}] {c['id']}")
            h = fetch_history(c["id"])
            if h is not None:
                hist[c["id"]] = h
            time.sleep(SLEEP_SECONDS)

    if "bitcoin" not in hist:
        raise RuntimeError("Bitcoin history missing, cannot continue.")
    btc_ret = hist["bitcoin"].pct_change().dropna()

    rows = []
    for c in coins:
        if c["id"] in hist:
            row = analyse_coin(c, hist[c["id"]], btc_ret)
            if row:
                rows.append(row)
    if not rows:
        raise RuntimeError("No coins could be analysed.")

    df = score_and_rank(pd.DataFrame(rows))
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(build_report(df, fg, demo))
    save_snapshot(df, fg, demo)
    print(f"Done. Wrote {REPORT_FILE} with {len(df)} coins.")


if __name__ == "__main__":
    main()

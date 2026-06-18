#!/usr/bin/env python3
"""
Module 9 - Demonstration 1: Building a Strands Agent for stocks research
========================================================================
A ready-to-run Strands Agent powered by Amazon Nova Lite on Amazon Bedrock.

The agent is given three custom tools (built on the free `yfinance` library, so
NO extra API key is needed) and reasons through which ones to call to answer a
stock-research question. This mirrors slide 11-13 of the instructor deck:
  Prompt -> Agent -> (Invoke model | Run tool) loop -> Result.

------------------------------------------------------------------------
SETUP (one time)
------------------------------------------------------------------------
    python -m venv .venv && source .venv/bin/activate      # optional
    pip install "strands-agents>=0.1" yfinance

    # AWS credentials (any standard method works):
    aws configure            # or set env vars / use an IAM role
    export AWS_REGION=us-east-1

    # In the Bedrock console, request model access for "Amazon Nova Lite".

------------------------------------------------------------------------
RUN
------------------------------------------------------------------------
    python demo1_strands_stocks_agent.py                   # runs a sample query
    python demo1_strands_stocks_agent.py "Compare AAPL and MSFT this year"
    python demo1_strands_stocks_agent.py --chat            # interactive mode
"""

import os
import sys

import yfinance as yf
from strands import Agent, tool
from strands.models import BedrockModel

# --------------------------------------------------------------------------- #
# Configuration (override with environment variables)
# --------------------------------------------------------------------------- #
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Nova requires a cross-region inference profile prefix: us. / eu. / global.
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")

SYSTEM_PROMPT = """You are a concise stock-research assistant.
Use the available tools to fetch real market data before answering — never
guess prices or figures. When comparing companies, call the tools for each
ticker. Summarise findings in plain language and always remind the user that
this is informational only and not financial advice."""


# --------------------------------------------------------------------------- #
# Tools  (the docstring + type hints become the tool description for the LLM)
# --------------------------------------------------------------------------- #
@tool
def get_stock_quote(ticker: str) -> str:
    """Get the latest price quote for a stock ticker (e.g. 'AAPL', 'TSLA').

    Returns the current/most-recent price, the previous close, the day's
    change, and the trading currency.
    """
    t = yf.Ticker(ticker)
    info = t.fast_info
    price = info.get("last_price")
    prev = info.get("previous_close")
    currency = info.get("currency", "USD")
    if price is None:
        return f"Could not find price data for '{ticker}'. Check the ticker symbol."
    change = price - prev if prev else 0.0
    pct = (change / prev * 100) if prev else 0.0
    return (
        f"{ticker.upper()}: {price:,.2f} {currency} "
        f"(prev close {prev:,.2f}, change {change:+,.2f} / {pct:+.2f}%)"
    )


@tool
def get_company_overview(ticker: str) -> str:
    """Get a company overview for a stock ticker: name, sector, industry,
    market capitalisation, and a short business summary.
    """
    t = yf.Ticker(ticker)
    info = t.info
    if not info or not info.get("longName"):
        return f"Could not find company information for '{ticker}'."
    cap = info.get("marketCap")
    cap_str = f"{cap / 1e9:,.1f}B" if cap else "n/a"
    summary = (info.get("longBusinessSummary") or "")[:400]
    return (
        f"Name: {info.get('longName')}\n"
        f"Sector: {info.get('sector', 'n/a')} | Industry: {info.get('industry', 'n/a')}\n"
        f"Market cap: {cap_str} | P/E (trailing): {info.get('trailingPE', 'n/a')}\n"
        f"Summary: {summary}..."
    )


@tool
def get_recent_performance(ticker: str, period: str = "6mo") -> str:
    """Get recent price performance for a ticker over a period.

    period options: '1mo', '3mo', '6mo', '1y', 'ytd', '5y'.
    Returns start price, end price, total return %, and the high/low over
    the window.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    if hist.empty:
        return f"No history found for '{ticker}' over period '{period}'."
    start = hist["Close"].iloc[0]
    end = hist["Close"].iloc[-1]
    ret = (end - start) / start * 100
    return (
        f"{ticker.upper()} over {period}: "
        f"start {start:,.2f}, end {end:,.2f}, total return {ret:+.2f}%, "
        f"high {hist['High'].max():,.2f}, low {hist['Low'].min():,.2f}"
    )


# --------------------------------------------------------------------------- #
# Agent factory
# --------------------------------------------------------------------------- #
def build_agent() -> Agent:
    model = BedrockModel(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        temperature=0.2,
    )
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_stock_quote, get_company_overview, get_recent_performance],
    )


def run_once(query: str) -> None:
    agent = build_agent()
    print(f"\n>>> {query}\n")
    agent(query)  # Strands streams the answer (incl. tool calls) to stdout
    print()


def run_chat() -> None:
    agent = build_agent()
    print("Strands stocks-research agent (Nova Lite). Type 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            agent(q)
            print()
    print("\nGoodbye!")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--chat":
        run_chat()
    elif args:
        run_once(" ".join(args))
    else:
        run_once(
            "Give me a quick research summary on NVIDIA (NVDA): current price, "
            "what the company does, and how it has performed over the last 6 months."
        )

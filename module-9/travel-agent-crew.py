#!/usr/bin/env python3
"""
Module 9 - Demonstration 2: Building a CrewAI agent for travel research
========================================================================
A ready-to-run CrewAI agent powered by Amazon Nova Lite on Amazon Bedrock.

The agent has a web-search tool (DuckDuckGo, NO API key required) and a single
task: research a travel destination and return a structured mini-guide. This
mirrors slides 14-16 of the instructor deck (Agent -> Task -> Crew -> kickoff).

------------------------------------------------------------------------
SETUP (one time)
------------------------------------------------------------------------
    python -m venv .venv && source .venv/bin/activate      # optional
    pip install "crewai>=0.80" ddgs

    # AWS credentials (any standard method works):
    aws configure            # or set env vars / use an IAM role
    export AWS_REGION=us-east-1

    # In the Bedrock console, request model access for "Amazon Nova Lite".

------------------------------------------------------------------------
RUN
------------------------------------------------------------------------
    python demo2_crewai_travel_agent.py                    # researches a sample city
    python demo2_crewai_travel_agent.py "Kyoto, Japan in autumn"
"""

import os
import sys

# Quieten CrewAI / LiteLLM noise before importing them.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

# --------------------------------------------------------------------------- #
# Configuration (override with environment variables)
# --------------------------------------------------------------------------- #
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# CrewAI uses LiteLLM for Bedrock; Nova needs the cross-region profile prefix.
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "bedrock/us.amazon.nova-lite-v1:0")


# --------------------------------------------------------------------------- #
# Web-search tool (DuckDuckGo — works without an API key)
# --------------------------------------------------------------------------- #
@tool("WebSearch")
def web_search(search_query: str) -> str:
    """Search the web for current information on a given topic and return the
    top results as title + snippet + URL."""
    try:
        from ddgs import DDGS  # current package name
    except ImportError:
        from duckduckgo_search import DDGS  # older package name fallback

    with DDGS() as ddgs:
        results = list(ddgs.text(search_query, max_results=6))
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}"
        for r in results
    )


# --------------------------------------------------------------------------- #
# LLM, Agent, Task, Crew
# --------------------------------------------------------------------------- #
def build_crew(destination: str) -> Crew:
    llm = LLM(
        model=MODEL_ID,
        aws_region_name=AWS_REGION,
        temperature=0.4,
        # Bedrock-friendly: let LiteLLM drop params a model doesn't accept.
        drop_params=True,
    )

    travel_agent = Agent(
        role="Travel Research Specialist",
        goal="Produce accurate, useful, up-to-date travel guidance for a destination",
        backstory=(
            "You are a seasoned travel researcher who blends web research with "
            "practical know-how. You always verify details with the search tool "
            "rather than relying on memory, and you write for a busy traveller."
        ),
        llm=llm,
        tools=[web_search],
        allow_delegation=False,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Research the travel destination: {destination}. "
            "Use the web search tool to gather current information. Cover: "
            "(1) best time to visit, (2) top 3-5 attractions, "
            "(3) typical local food to try, (4) one practical tip "
            "(getting around, budget, or safety), and (5) a rough idea of costs."
        ),
        expected_output=(
            "A concise markdown mini-guide with five clearly labelled sections: "
            "Best time to visit, Top attractions, Local food, Practical tip, "
            "Approximate costs."
        ),
        agent=travel_agent,
    )

    return Crew(
        agents=[travel_agent],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    destination = " ".join(sys.argv[1:]) or "Lisbon, Portugal for a long weekend"
    crew = build_crew(destination)
    result = crew.kickoff()
    print("\n" + "=" * 70)
    print("TRAVEL GUIDE")
    print("=" * 70)
    print(result)

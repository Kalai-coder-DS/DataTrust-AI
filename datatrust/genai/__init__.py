"""
Phase 9: plain-language summaries of an analysis.

Three providers behind one interface (providers.py): a template (default, offline, deterministic),
a local model through Ollama, and the Claude API. All receive the same verified facts (facts.py),
AI output is checked against them (guard.py), and any failure falls back to the template
(summary.py). Neither Ollama nor the Claude API is required; no new Python packages are needed.
"""
from datatrust.genai.facts import build_facts
from datatrust.genai.summary import PROVIDERS, summarize

__all__ = ["PROVIDERS", "build_facts", "summarize"]

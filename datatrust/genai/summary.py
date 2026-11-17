"""
datatrust/genai/summary.py

The one function the rest of the project calls:

    summary = summarize(analysis, provider="auto")

Choosing a provider (setting `provider`, or DATATRUST_SUMMARY_PROVIDER):
    template   always the template text
    ollama     the local model; falls back to the template if it is not running or fails
    claude     the Claude API; falls back to the template if no key is set or it fails
    auto       Claude if ANTHROPIC_API_KEY is set, else Ollama if it is running with the model,
               else the template

Every AI text goes through guard.check_summary; a rejected text also falls back to the template.
The result says what happened, so the report can show it honestly:

    {"text": "...", "provider": "template", "requested": "claude", "model": None,
     "fallback_reason": "Claude API: HTTP 401 ...", "facts": {...}}

Configuration (environment variables or .env; never hard-coded):
    ANTHROPIC_API_KEY          enables the Claude provider
    DATATRUST_CLAUDE_MODEL     default claude-haiku-5-5 (fast, low cost)
    DATATRUST_OLLAMA_URL       default http://localhost:11434
    DATATRUST_OLLAMA_MODEL     default llama3.2:3b (about 2 GB; runs on a 16 GB laptop without a GPU)
    DATATRUST_AI_TIMEOUT       seconds to wait for an AI answer, default 120 (a CPU-only laptop is slow)
"""
from __future__ import annotations

import os

from datatrust.genai.facts import build_facts
from datatrust.genai.guard import SummaryRejected, check_summary
from datatrust.genai.providers import (
    ClaudeProvider,
    OllamaProvider,
    ProviderError,
    TemplateProvider,
    Transport,
    http_json,
)
from datatrust.logger import get_logger

logger = get_logger(__name__)

PROVIDERS = ("auto", "template", "ollama", "claude")
DEFAULTS = {
    "DATATRUST_CLAUDE_MODEL": "claude-haiku-5-5",
    "DATATRUST_OLLAMA_URL": "http://localhost:11434",
    "DATATRUST_OLLAMA_MODEL": "llama3.2:3b",
    "DATATRUST_AI_TIMEOUT": "120",
}
LABELS = {"template": "Template summary (offline)", "ollama": "Local AI (Ollama)", "claude": "Claude API"}


def _environ(environ: dict | None) -> dict:
    if environ is not None:
        return environ
    from datatrust.storage import load_env_file      # .env support, same file as the database settings
    load_env_file()
    return os.environ


def _setting(env: dict, key: str) -> str:
    return (env.get(key) or DEFAULTS.get(key, "")).strip()


def build_providers(environ: dict | None = None, transport: Transport = http_json) -> dict:
    env = _environ(environ)
    try:
        timeout = float(_setting(env, "DATATRUST_AI_TIMEOUT"))
    except ValueError:
        timeout = float(DEFAULTS["DATATRUST_AI_TIMEOUT"])
    return {
        "template": TemplateProvider(),
        "ollama": OllamaProvider(_setting(env, "DATATRUST_OLLAMA_URL"), _setting(env, "DATATRUST_OLLAMA_MODEL"),
                                 timeout, transport),
        "claude": ClaudeProvider((env.get("ANTHROPIC_API_KEY") or "").strip() or None,
                                 _setting(env, "DATATRUST_CLAUDE_MODEL"), timeout, transport),
    }


def requested_provider(provider: str | None, environ: dict | None = None) -> str:
    choice = (provider or _environ(environ).get("DATATRUST_SUMMARY_PROVIDER") or "auto").strip().lower()
    if choice not in PROVIDERS:
        raise ValueError(f"Unknown summary provider '{choice}'. Choose one of: {', '.join(PROVIDERS)}.")
    return choice


def summarize(analysis: dict, provider: str | None = None, environ: dict | None = None,
              transport: Transport = http_json) -> dict:
    """Never raises for provider problems: the worst case is the template summary with a reason."""
    facts = build_facts(analysis)
    requested = requested_provider(provider, environ)
    providers = build_providers(environ, transport)

    if requested == "auto":
        if providers["claude"].available():
            chosen = "claude"
        elif providers["ollama"].available():
            chosen = "ollama"
        else:
            chosen = "template"
    else:
        chosen = requested

    reason = None
    if chosen != "template":
        p = providers[chosen]
        try:
            if chosen == "claude" and not p.available():
                raise ProviderError("ANTHROPIC_API_KEY is not set")
            text = check_summary(p.summarize(facts), facts)
            logger.info("Summary written by %s (%s)", chosen, p.model)
            return {"text": text, "provider": chosen, "label": LABELS[chosen], "requested": requested,
                    "model": p.model, "fallback_reason": None, "facts": facts}
        except (ProviderError, SummaryRejected) as exc:
            reason = f"{LABELS[chosen]}: {exc}"
            logger.warning("Summary: %s; using the template summary instead", reason)

    return {"text": TemplateProvider().summarize(facts), "provider": "template", "label": LABELS["template"],
            "requested": requested, "model": None, "fallback_reason": reason, "facts": facts}

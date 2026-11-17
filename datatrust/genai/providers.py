"""
datatrust/genai/providers.py

Three providers, one interface:

    class SummaryProvider:
        name: str
        def available(self) -> bool            # cheap check: key present / server reachable
        def summarize(self, facts) -> str      # plain text; raises ProviderError on failure

  TemplateProvider  always available, offline, deterministic (same facts -> same text).
  OllamaProvider    a local model through Ollama's HTTP API (http://localhost:11434). Free, data
                    stays on the computer. Optional: nothing to install unless you want it.
  ClaudeProvider    the Claude API. Only available when ANTHROPIC_API_KEY is set. Optional.

No new Python packages: HTTP goes through urllib (standard library). The HTTP call is injected
(`transport`) so tests run offline with a fake and never touch the network.
API keys come only from the environment and never appear in results, logs or error messages.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Protocol

Transport = Callable[[str, dict, dict, float], dict]     # (url, json_body or {}, headers, timeout) -> json


class ProviderError(Exception):
    """A provider could not produce a summary. The message is safe to show (no secrets)."""


def http_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    """POST body as JSON (GET when body is empty) and return the JSON answer."""
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, method="POST" if body else "GET",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", "")
            detail = detail.get("message", "") if isinstance(detail, dict) else str(detail)
        except Exception:
            pass
        raise ProviderError(f"HTTP {exc.code} from {url.split('/v1')[0].split('/api')[0]}"
                            + (f": {detail[:200]}" if detail else "")) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ProviderError(f"cannot reach {url.split('/v1')[0].split('/api')[0]} ({reason})") from None
    except json.JSONDecodeError:
        raise ProviderError("the answer was not valid JSON") from None


SYSTEM_PROMPT = """You explain a data-quality report to a busy manager who is not a data scientist.
You receive VERIFIED FACTS as JSON, produced by an analysis program. Rules:
1. Use ONLY the facts. Do not guess, add, round differently, or calculate any number.
   Copy scores and counts exactly as written in the facts, or leave them out.
2. Never give your own score or grade. The Data Trust and ML Readiness scores are final.
3. Write 120-200 words of plain text in 2-3 short paragraphs. No headings, no lists, no markdown.
4. Say: whether the data can be trusted and used for a model, the most important problems,
   and what to fix first and why. If the facts include a model section, say what it showed.
5. Describe a grade with its "grade_means" words (e.g. grade C = "usable with fixes").
6. Counts: the total number of issues is issue_counts.total. Call issues "critical" or "high" only
   if issue_counts.critical or issue_counts.high is above 0.
7. Model: say "the model scores <after_cleaning> against <guessing> for guessing, beating it by
   <beats_guessing_by>". Never call beats_guessing_by an accuracy."""


def user_prompt(facts: dict) -> str:
    return "VERIFIED FACTS:\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n\nWrite the summary."


class SummaryProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def summarize(self, facts: dict) -> str: ...


# --------------------------------------------------------------------------- template
GRADE_WORDS = {"A": "excellent", "B": "good", "C": "usable with fixes", "D": "poor", "F": "not usable as-is"}


class TemplateProvider:
    name = "template"

    def available(self) -> bool:
        return True

    def summarize(self, facts: dict) -> str:
        t, m = facts["data_trust"], facts["ml_readiness"]
        c = facts["issue_counts"]
        p1 = (f"'{facts['dataset']}' ({facts['rows']:,} rows, {facts['columns']} columns) has a Data Trust Score of "
              f"{t['score']} (grade {t['grade']}, {GRADE_WORDS.get(t['grade'], '')}) and an ML Readiness Score of "
              f"{m['score']} (grade {m['grade']}, {GRADE_WORDS.get(m['grade'], '')})")
        p1 += f" for predicting '{facts['target']}'." if facts["target"] else "; no target column was given, so the ML assessment is partial."
        serious = c["critical"] + c["high"]
        if c["total"] == 0:
            p1 += " The check found no issues."
        elif serious:
            p1 += f" The check found {c['total']} issue(s), {serious} of them critical or high."
        else:
            p1 += f" The check found {c['total']} issue(s), none critical or high."
        paras = [p1]
        if facts["top_actions"]:
            first = facts["top_actions"][0]
            p2 = f"Fix first: {first['problem'].split('] ', 1)[-1].rstrip('.')}. {first['action']}"
            if len(facts["top_actions"]) > 1:
                p2 += f" The report lists {len(facts['top_actions'])} actions in order of benefit."
            paras.append(p2)
        model = facts.get("model")
        if model:
            lines = list(model.get("verdict", [])[:2]) + list(model.get("importance_verdict", [])[:2])
            if lines:
                paras.append(" ".join(lines))
        return "\n\n".join(paras)


# --------------------------------------------------------------------------- Ollama
class OllamaProvider:
    name = "ollama"

    def __init__(self, url: str, model: str, timeout: float, transport: Transport = http_json):
        self.url, self.model, self.timeout, self.transport = url.rstrip("/"), model, timeout, transport

    def available(self) -> bool:
        """Server running AND the configured model downloaded (2-second check)."""
        try:
            tags = self.transport(f"{self.url}/api/tags", {}, {}, 2.0)
        except ProviderError:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def summarize(self, facts: dict) -> str:
        body = {"model": self.model, "stream": False,
                "options": {"temperature": 0.2, "num_predict": 400},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user_prompt(facts)}]}
        answer = self.transport(f"{self.url}/api/chat", body, {}, self.timeout)
        text = (answer.get("message") or {}).get("content", "")
        if not text:
            raise ProviderError(f"Ollama returned no text ({answer.get('error', 'empty answer')})")
        return text


# --------------------------------------------------------------------------- Claude
class ClaudeProvider:
    name = "claude"
    URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str | None, model: str, timeout: float, transport: Transport = http_json):
        self._key, self.model, self.timeout, self.transport = api_key, model, timeout, transport

    def __repr__(self) -> str:                       # never print the key
        return f"ClaudeProvider(model={self.model!r}, key={'set' if self._key else 'missing'})"

    def available(self) -> bool:
        return bool(self._key)

    def summarize(self, facts: dict) -> str:
        if not self._key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        body = {"model": self.model, "max_tokens": 600, "temperature": 0.2, "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt(facts)}]}
        headers = {"x-api-key": self._key, "anthropic-version": self.API_VERSION}
        try:
            answer = self.transport(self.URL, body, headers, self.timeout)
        except ProviderError as exc:
            raise ProviderError(str(exc).replace(self._key, "***")) from None
        text = "".join(b.get("text", "") for b in answer.get("content", []) if b.get("type") == "text")
        if not text:
            raise ProviderError("the Claude API returned no text")
        return text

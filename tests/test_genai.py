"""
tests/test_genai.py

Phase 9 summaries. Fully OFFLINE: the network is blocked for this whole file (any real HTTP call
fails the test), Ollama and the Claude API are replaced by a fake transport, and every test passes
its own environment instead of reading your real .env or ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
import urllib.request

import pytest

from datatrust.genai import build_facts, summarize
from datatrust.genai.guard import SummaryRejected, check_summary
from datatrust.genai.present import summary_view
from datatrust.genai.providers import (
    ClaudeProvider,
    OllamaProvider,
    ProviderError,
    TemplateProvider,
    http_json,
)
from datatrust.genai.summary import build_providers, requested_provider
from datatrust.pipeline import analyze_file
from datatrust.reporting import render_html

KEY = "sk-ant-test-0000-SECRET"
OLLAMA_TAGS = {"models": [{"name": "llama3.2:3b"}]}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("a test tried to use the real network")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


@pytest.fixture(scope="module")
def analysis(sample_dir):
    return analyze_file(sample_dir / "customers_dirty.csv", target="churned", baseline=True)


class FakeTransport:
    """Answers like Ollama / the Claude API would, and records every call."""

    def __init__(self, ollama_up=True, ollama_text="", claude_text="", fail=None):
        self.calls, self.ollama_up, self.ollama_text, self.claude_text, self.fail = [], ollama_up, ollama_text, claude_text, fail

    def __call__(self, url, body, headers, timeout):
        self.calls.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        if self.fail and self.fail in url:
            raise ProviderError(f"cannot reach {url} (connection refused)")
        if url.endswith("/api/tags"):
            if not self.ollama_up:
                raise ProviderError("cannot reach http://localhost:11434 (connection refused)")
            return OLLAMA_TAGS
        if url.endswith("/api/chat"):
            return {"message": {"role": "assistant", "content": self.ollama_text}}
        if "anthropic" in url:
            return {"content": [{"type": "text", "text": self.claude_text}]}
        raise AssertionError(url)


def good_text(facts):
    t, m = facts["data_trust"], facts["ml_readiness"]
    return (f"The file has a Data Trust Score of {t['score']} and an ML Readiness Score of {m['score']}. "
            f"It has {facts['rows']} rows. Fix the placeholders in 'gender' first.")


# ------------------------------------------------------------------ facts
def test_facts_are_copied_from_the_analysis_not_recalculated(analysis):
    f = build_facts(analysis)
    assert f["data_trust"]["score"] == analysis["scores"]["trust"]["score"] == 79.8
    assert f["ml_readiness"]["score"] == analysis["scores"]["ml_readiness"]["score"]
    assert f["rows"] == 515 and f["target"] == "churned"
    assert f["issue_counts"]["total"] == analysis["quality"]["total_issues"]
    assert f["model"]["relied_on"] == ["tenure_months"] and f["model"]["after_cleaning"] == analysis["ml"]["cleaned"]["score"]
    json.dumps(f)                                         # what is sent must be plain JSON


def test_facts_without_model_or_target(sample_dir):
    f = build_facts(analyze_file(sample_dir / "customers_clean.csv"))
    assert f["target"] is None and "model" not in f


# ------------------------------------------------------------------ guard
def test_guard_accepts_numbers_from_the_facts(analysis):
    f = build_facts(analysis)
    assert check_summary(good_text(f), f)
    assert check_summary("The top 3 problems are listed out of 100.", f)


@pytest.mark.parametrize("bad, why", [
    ("The Data Trust Score is 85.0.", "85"),                     # invented score
    ("About 42% of rows are broken.", "42"),                    # invented percentage
    ("", "empty"),
    ("x" * 5000, "too long"),
])
def test_guard_rejects_invented_numbers_and_junk(analysis, bad, why):
    with pytest.raises(SummaryRejected, match=why):
        check_summary(bad, build_facts(analysis))


def test_template_text_always_passes_its_own_guard(analysis, sample_dir):
    for a in (analysis, analyze_file(sample_dir / "customers_clean.csv")):
        f = build_facts(a)
        assert check_summary(TemplateProvider().summarize(f), f)


def test_template_is_deterministic(analysis):
    f = build_facts(analysis)
    assert TemplateProvider().summarize(f) == TemplateProvider().summarize(f)


# ------------------------------------------------------------------ selection
def test_auto_without_key_or_ollama_uses_template(analysis):
    t = FakeTransport(ollama_up=False)
    r = summarize(analysis, "auto", environ={}, transport=t)
    assert (r["provider"], r["requested"], r["fallback_reason"]) == ("template", "auto", None)
    assert [c["url"] for c in t.calls] == ["http://localhost:11434/api/tags"]   # only the 2-second check


def test_auto_prefers_claude_when_key_is_set(analysis):
    f = build_facts(analysis)
    t = FakeTransport(claude_text=good_text(f))
    r = summarize(analysis, "auto", environ={"ANTHROPIC_API_KEY": KEY}, transport=t)
    assert r["provider"] == "claude" and r["model"] == "claude-haiku-5-5" and r["fallback_reason"] is None
    call = t.calls[-1]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == KEY and call["headers"]["anthropic-version"] == "2023-06-01"
    assert json.dumps(f, indent=1, ensure_ascii=False) in call["body"]["messages"][0]["content"]


def test_auto_uses_ollama_when_running_and_no_key(analysis):
    t = FakeTransport(ollama_text=good_text(build_facts(analysis)))
    r = summarize(analysis, "auto", environ={}, transport=t)
    assert r["provider"] == "ollama" and r["model"] == "llama3.2:3b"
    assert t.calls[-1]["url"] == "http://localhost:11434/api/chat" and t.calls[-1]["body"]["stream"] is False


def test_auto_skips_ollama_when_model_not_downloaded(analysis):
    t = FakeTransport(ollama_text="unused")
    r = summarize(analysis, "auto", environ={"DATATRUST_OLLAMA_MODEL": "qwen2.5:3b"}, transport=t)
    assert r["provider"] == "template"


def test_all_providers_receive_the_same_facts(analysis):
    f = build_facts(analysis)
    t = FakeTransport(ollama_text=good_text(f), claude_text=good_text(f))
    summarize(analysis, "ollama", environ={}, transport=t)
    summarize(analysis, "claude", environ={"ANTHROPIC_API_KEY": KEY}, transport=t)
    ollama_prompt = t.calls[0]["body"]["messages"][1]["content"]
    claude_prompt = t.calls[1]["body"]["messages"][0]["content"]
    assert ollama_prompt == claude_prompt


def test_configuration_comes_from_the_environment():
    p = build_providers({"DATATRUST_OLLAMA_URL": "http://box:9999/", "DATATRUST_OLLAMA_MODEL": "gemma3:4b",
                         "DATATRUST_CLAUDE_MODEL": "claude-sonnet-5-5", "DATATRUST_AI_TIMEOUT": "30",
                         "ANTHROPIC_API_KEY": KEY}, transport=FakeTransport())
    assert (p["ollama"].url, p["ollama"].model, p["ollama"].timeout) == ("http://box:9999", "gemma3:4b", 30.0)
    assert p["claude"].model == "claude-sonnet-5-5" and p["claude"].available()
    assert build_providers({"DATATRUST_AI_TIMEOUT": "soon"}, transport=FakeTransport())["ollama"].timeout == 120.0


def test_provider_from_environment_and_invalid_choice():
    assert requested_provider(None, {"DATATRUST_SUMMARY_PROVIDER": "Template"}) == "template"
    assert requested_provider(None, {}) == "auto"
    with pytest.raises(ValueError, match="Unknown summary provider 'gpt'"):
        requested_provider("gpt", {})


# ------------------------------------------------------------------ fallback
def test_claude_requested_without_key_falls_back(analysis):
    t = FakeTransport()
    r = summarize(analysis, "claude", environ={}, transport=t)
    assert r["provider"] == "template" and r["requested"] == "claude"
    assert r["fallback_reason"] == "Claude API: ANTHROPIC_API_KEY is not set"
    assert t.calls == []                                       # nothing was sent anywhere


def test_claude_failure_falls_back_and_hides_the_key(analysis):
    def failing(url, body, headers, timeout):
        raise ProviderError(f"HTTP 401 from https://api.anthropic.com: invalid x-api-key {KEY}")
    r = summarize(analysis, "claude", environ={"ANTHROPIC_API_KEY": KEY}, transport=failing)
    assert r["provider"] == "template" and "HTTP 401" in r["fallback_reason"]
    assert KEY not in json.dumps(r) and "***" in r["fallback_reason"]
    assert KEY not in repr(ClaudeProvider(KEY, "m", 5, failing))


def test_ollama_not_running_falls_back(analysis):
    r = summarize(analysis, "ollama", environ={}, transport=FakeTransport(fail="/api/chat"))
    assert r["provider"] == "template" and r["fallback_reason"].startswith("Local AI (Ollama): cannot reach")


def test_ollama_empty_answer_falls_back(analysis):
    r = summarize(analysis, "ollama", environ={}, transport=FakeTransport(ollama_text=""))
    assert r["provider"] == "template" and "no text" in r["fallback_reason"]


def test_invented_number_from_ai_falls_back(analysis):
    t = FakeTransport(claude_text="Data Trust is 91.5 and ML Readiness 88.")
    r = summarize(analysis, "claude", environ={"ANTHROPIC_API_KEY": KEY}, transport=t)
    assert r["provider"] == "template" and "not in the verified facts: 88, 91.5" in r["fallback_reason"]
    assert "91.5" not in r["text"]


def test_unexpected_answer_shape_falls_back(analysis):
    r = summarize(analysis, "claude", environ={"ANTHROPIC_API_KEY": KEY},
                  transport=lambda *a: {"content": [{"type": "tool_use"}]})
    assert r["provider"] == "template" and "no text" in r["fallback_reason"]


def test_real_http_errors_become_provider_errors(monkeypatch):
    import urllib.error
    def refuse(*a, **k):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(ProviderError, match=r"cannot reach http://localhost:11434 \(connection refused\)"):
        http_json("http://localhost:11434/api/tags", {}, {}, 1)
    assert OllamaProvider("http://localhost:11434", "llama3.2:3b", 1).available() is False


# ------------------------------------------------------------------ integration
def test_pipeline_report_and_storage(sample_dir, tmp_path):
    plain = analyze_file(sample_dir / "customers_clean.csv")
    assert "summary" not in plain and "Summary <span" not in render_html(plain)   # earlier phases unchanged
    a = analyze_file(sample_dir / "customers_dirty.csv", target="churned", summary="template")
    assert a["summary"]["provider"] == "template"
    html = render_html(a)
    assert "Summary <span" in html and "Template summary (offline)" in html
    from datatrust.storage import AnalysisRepository, Database, DBConfig
    db = Database(DBConfig(backend="sqlite", sqlite_path=tmp_path / "s.db")).connect()
    repo = AnalysisRepository(db)
    repo.init_schema()
    assert repo.get_analysis(repo.save(a))["summary"]["text"] == a["summary"]["text"]
    db.close()


def test_view_shows_fallback_honestly():
    v = summary_view({"text": "One.\n\nTwo.", "provider": "template", "label": "Template summary (offline)",
                      "model": None, "fallback_reason": "Claude API: ANTHROPIC_API_KEY is not set"})
    assert v["paragraphs"] == ["One.", "Two."] and not v["is_ai"]
    assert v["fallback"] == "Claude API: ANTHROPIC_API_KEY is not set. The template summary was used instead."
    assert summary_view(None) is None


def test_report_escapes_summary_text(analysis):
    html = render_html({**analysis, "summary": {"text": "<script>alert(1)</script>", "provider": "claude",
                                                "label": "Claude API", "model": "m", "fallback_reason": None}})
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_cli_summary(sample_dir, capsys):
    from main import run
    assert run([str(sample_dir / "customers_clean.csv"), "--summary", "template"]) == 0
    assert "PLAIN-LANGUAGE SUMMARY: Template summary (offline)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run([str(sample_dir / "customers_clean.csv"), "--summary", "gpt"])


# ------------------------------------------------------------------ Phase 9 fix: meaning checks
# Written by llama3.2:3b through Ollama on the sample file (Windows, 17 Nov 2026). It passed the
# first guard because every number in it also exists somewhere in the facts.
REAL_OLLAMA_TEXT = (
    "The ML readiness score is also 79.8, with a grade of \"C\", indicating that the data is suitable for "
    "machine learning models. However, there are some issues with data quality, such as duplicate rows and "
    "placeholder values in the \"gender\" column.\n\n"
    "The top issues list highlights seven problems, including missing values, non-numeric values, and "
    "duplicate rows.\n\n"
    "The model section shows that the model beats guessing by a balanced accuracy of 0.693, indicating that "
    "the features do carry signal about the \"churned\" target.\n\n"
    "Overall, the data quality report suggests that the dataset has some issues that need to be addressed "
    "before it can be used for machine learning models. The most critical problems are related to data quality."
)


def test_real_ollama_mistakes_are_now_rejected(analysis):
    f = build_facts(analysis)
    with pytest.raises(SummaryRejected, match="'seven problems', but the issue counts are 12 in total"):
        check_summary(REAL_OLLAMA_TEXT, f)


@pytest.mark.parametrize("bad, why", [
    ("The report found 7 issues.", "issue counts are 12"),
    ("The model beats guessing by a balanced accuracy of 0.693.", "beats guessing by 0.193"),
    ("The most critical problems are in 'gender'.", "no issue is critical"),
    ("There are high-severity issues in 'age'.", "no issue is high"),
])
def test_each_meaning_check(analysis, bad, why):
    with pytest.raises(SummaryRejected, match=why):
        check_summary(bad, build_facts(analysis))


def test_correct_phrasing_passes(analysis):
    f = build_facts(analysis)
    ok = ("The check found twelve issues, 9 medium and 3 low; the report lists 8 problems below. "
          "The model scores 0.693 against 0.5 for guessing, beating it by 0.193. "
          "Grade C means usable with fixes.")
    assert check_summary(ok, f) == ok
    assert check_summary("The model beats guessing by 0.193.", f)


def test_facts_carry_the_values_the_prompt_asks_for(analysis):
    f = build_facts(analysis)
    assert f["data_trust"]["grade_means"] == "usable with fixes"
    assert f["issues_listed_below"] == 8 and f["model"]["beats_guessing_by"] == analysis["ml"]["cleaned"]["lift"]


def test_real_mistakes_fall_back_to_the_template(analysis):
    r = summarize(analysis, "ollama", environ={}, transport=FakeTransport(ollama_text=REAL_OLLAMA_TEXT))
    assert r["provider"] == "template" and "'seven problems'" in r["fallback_reason"]

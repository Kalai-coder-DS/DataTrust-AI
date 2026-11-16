"""
tests/test_webapp.py

Phase 6 Django app, two layers of checks:

1. WITHOUT Django (always runs): the templates are checked against REAL page data built by
   datatrust.web. Django silently renders a misspelt variable as an empty string, so a typo
   like {{ latest.trst_score }} would never fail a page test that only checks status 200.
   Here every {{ variable.path }}, loop variable, {% with %}, {% include %} and {% url %}
   (name and number of arguments) is resolved against the data, tags must balance, and only
   known filters may be used.

2. WITH Django installed: `manage.py check` and the full Django test suite
   (webapp/dashboard/tests.py: every page through the Django test client) run as part of
   `python -m pytest`. Skipped, with a reason, if Django is not installed.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from datatrust.storage import AnalysisRepository, Database, DBConfig
from datatrust.web import analyze_upload, dashboard_page, dataset_page, run_page

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = ROOT / "webapp"
TEMPLATES = WEBAPP / "dashboard" / "templates"

BLOCK_TAGS = {"block", "for", "if", "with", "comment", "spaceless", "autoescape", "filter", "verbatim"}
INNER_TAGS = {"else", "elif", "empty"}
KNOWN_TAGS = BLOCK_TAGS | INNER_TAGS | {f"end{t}" for t in BLOCK_TAGS} | {
    "extends", "include", "load", "url", "csrf_token"}
KNOWN_FILTERS = {"floatformat", "length", "pluralize", "default", "safe", "intcomma", "stringformat"}
HUMANIZE_FILTERS = {"intcomma"}
OPAQUE = {"request", "form", "forloop", "csrf_token"}      # Django objects, not our data
IF_WORDS = {"not", "and", "or", "in", "is", "None", "True", "False"}
TOKEN = re.compile(r"({%.*?%}|{{.*?}})", re.S)
MISSING = object()


# ------------------------------------------------------------------ helpers
def template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def all_templates() -> list[str]:
    return sorted(str(p.relative_to(TEMPLATES)).replace("\\", "/") for p in TEMPLATES.rglob("*.html"))


def url_routes() -> dict[str, int]:
    """{'dataset': 1, ...}: URL name -> number of arguments, read from dashboard/urls.py."""
    tree = ast.parse((WEBAPP / "dashboard" / "urls.py").read_text())
    routes = {}
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "path":
            route = call.args[0].value
            name = next(k.value.value for k in call.keywords if k.arg == "name")
            routes[name] = route.count("<")
    return routes


def is_literal(token: str) -> bool:
    return bool(re.fullmatch(r'"[^"]*"|\'[^\']*\'|-?\d+(\.\d+)?', token))


def split_filters(expr: str) -> tuple[str, list[tuple[str, str | None]]]:
    parts = [p.strip() for p in re.split(r"\|(?=(?:[^\"']|\"[^\"]*\"|'[^']*')*$)", expr)]
    filters = []
    for f in parts[1:]:
        name, _, arg = f.partition(":")
        filters.append((name.strip(), arg.strip() or None))
    return parts[0], filters


class TemplateChecker:
    """Walks a template like Django would, resolving every variable against real data."""

    def __init__(self, routes: dict[str, int]):
        self.routes = routes
        self.errors: list[str] = []

    def resolve(self, path: str, scopes: list[dict], where: str):
        if is_literal(path):
            return None
        root, *attrs = path.split(".")
        for scope in reversed(scopes):
            if root in scope:
                value = scope[root]
                break
        else:
            if root in OPAQUE:
                return MISSING
            self.errors.append(f"{where}: unknown variable '{root}' in '{path}'")
            return MISSING
        if value is MISSING or root in OPAQUE:
            return MISSING
        for i, attr in enumerate(attrs):
            if isinstance(value, dict) and attr in value:
                value = value[attr]
            elif isinstance(value, (list, tuple)) and attr.isdigit() and int(attr) < len(value):
                value = value[int(attr)]
            else:
                self.errors.append(f"{where}: '{'.'.join([root, *attrs[:i + 1]])}' does not exist in the page data")
                return MISSING
        return value

    def check_expr(self, expr: str, scopes, where, loads):
        var, filters = split_filters(expr)
        self.resolve(var, scopes, where)
        for name, arg in filters:
            if name not in KNOWN_FILTERS:
                self.errors.append(f"{where}: unknown filter '{name}'")
            if name in HUMANIZE_FILTERS and "humanize" not in loads:
                self.errors.append(f"{where}: '{name}' needs {{% load humanize %}}")
            if arg and not is_literal(arg):
                self.resolve(arg, scopes, where)

    def check(self, name: str, context: dict, scopes: list[dict] | None = None):
        text = template(name)
        scopes = [context] if scopes is None else scopes
        stack: list[tuple[str, bool]] = []          # (tag, pushed a scope?)
        loads: set[str] = set()
        tokens = TOKEN.findall(text)
        # Every opener must start a complete tag ('}}' alone is fine: CSS ends media queries with it).
        if text.count("{{") != sum(t.startswith("{{") for t in tokens) or \
                text.count("{%") != sum(t.startswith("{%") for t in tokens):
            self.errors.append(f"{name}: a {{{{ or {{% is never closed")
        for tok in tokens:
            where = f"{name}: {tok[:60]}"
            if tok.startswith("{{"):
                self.check_expr(tok[2:-2].strip(), scopes, where, loads)
                continue
            body = tok[2:-2].strip()
            tag, _, rest = body.partition(" ")
            rest = rest.strip()
            if tag not in KNOWN_TAGS:
                self.errors.append(f"{where}: unknown tag '{tag}'")
            if tag == "load":
                loads.update(rest.split())
            elif tag == "extends" or tag == "include":
                target = rest.strip("\"'")
                if not (TEMPLATES / target).is_file():
                    self.errors.append(f"{where}: template '{target}' does not exist")
                elif tag == "include":
                    self.check(target, context, scopes)
            elif tag == "url":
                parts = rest.split()
                url_name = parts[0].strip("\"'")
                ns, _, short = url_name.partition(":")
                if ns != "dashboard" or short not in self.routes:
                    self.errors.append(f"{where}: unknown URL name '{url_name}'")
                elif len(parts) - 1 != self.routes[short]:
                    self.errors.append(f"{where}: '{url_name}' needs {self.routes[short]} argument(s), got {len(parts) - 1}")
                for arg in parts[1:]:
                    self.resolve(arg, scopes, where)
            elif tag in ("if", "elif"):
                for word in re.split(r"\s+|==|!=|<=|>=|<|>|\(|\)", rest):
                    if word and word not in IF_WORDS:
                        self.check_expr(word, scopes, where, loads)
                if tag == "if":
                    stack.append(("if", False))
            elif tag == "for":
                m = re.fullmatch(r"(\w+) in (.+)", rest)
                if not m:
                    self.errors.append(f"{where}: cannot parse for-loop")
                    stack.append(("for", False))
                    continue
                seq = self.resolve(split_filters(m.group(2))[0], scopes, where)
                first = seq[0] if isinstance(seq, (list, tuple)) and seq else MISSING
                scopes.append({m.group(1): first})
                stack.append(("for", True))
            elif tag == "with":
                new = {}
                for pair in rest.split():
                    k, _, v = pair.partition("=")
                    new[k] = self.resolve(v, scopes, where)
                scopes.append(new)
                stack.append(("with", True))
            elif tag in BLOCK_TAGS:
                stack.append((tag, False))
            elif tag.startswith("end"):
                if not stack or stack[-1][0] != tag[3:]:
                    self.errors.append(f"{where}: '{tag}' does not close '{stack[-1][0] if stack else 'nothing'}'")
                    continue
                _, pushed = stack.pop()
                if pushed:
                    scopes.pop()
        if stack:
            self.errors.append(f"{name}: unclosed {[t for t, _ in stack]}")


# ------------------------------------------------------------------ real page data
@pytest.fixture(scope="module")
def pages(sample_dir, tmp_path_factory):
    db = Database(DBConfig(backend="sqlite", sqlite_path=tmp_path_factory.mktemp("tpl") / "t.db")).connect()
    repo = AnalysisRepository(db)
    repo.init_schema()
    dirty = (sample_dir / "customers_dirty.csv").read_bytes()
    analyze_upload([dirty], "customers_dirty.csv", repo, target="churned")
    run_id = analyze_upload([dirty], "customers_dirty.csv", repo, target="churned", baseline=True)
    data = {
        "dashboard/home.html": dashboard_page(repo),
        "dashboard/dataset.html": dataset_page(repo, "customers_dirty.csv"),
        "dashboard/run.html": run_page(repo, run_id),
        "dashboard/upload.html": {"form": object(), "limit_mb": 200},
        "dashboard/error.html": {"status": 404, "title": "Not found", "message": "x"},
    }
    db.close()
    return data


# ------------------------------------------------------------------ checks without Django
def test_every_page_template_resolves_against_real_data(pages):
    checker = TemplateChecker(url_routes())
    for name, context in pages.items():
        checker.check(name, dict(context))
    checker.check("dashboard/base.html", {})
    assert checker.errors == []


def test_checker_catches_typos(tmp_path, monkeypatch, pages):
    """The checker itself must fail on the mistakes it is meant to catch."""
    bad = tmp_path / "dashboard"
    bad.mkdir()
    (bad / "bad.html").write_text(
        "{% for r in runs %}{{ r.trst_score }}{% endfor %}{{ nme }}{{ name|floatfmt:1 }}"
        "{% url 'dashboard:run' %}{% if latest %}{{ latest.n_rows|intcomma }}", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "TEMPLATES", tmp_path)
    checker = TemplateChecker(url_routes())
    checker.check("dashboard/bad.html", dict(pages["dashboard/dataset.html"]))
    text = "\n".join(checker.errors)
    for expected in ["r.trst_score", "unknown variable 'nme'", "unknown filter 'floatfmt'",
                     "needs 1 argument(s), got 0", "needs {% load humanize %}", "unclosed ['if']"]:
        assert expected in text, (expected, text)


def test_every_template_is_used_and_exists():
    views = (WEBAPP / "dashboard" / "views.py").read_text()
    used = set(re.findall(r'"(dashboard/[\w.]+\.html)"', views))
    for name in used:
        assert (TEMPLATES / name).is_file(), name
    included = {m for t in all_templates() for m in re.findall(r'{% (?:include|extends) "([^"]+)"', template(t))}
    assert set(all_templates()) == used | included


def test_every_view_in_urls_exists_and_is_wrapped():
    views = (WEBAPP / "dashboard" / "views.py").read_text()
    tree = ast.parse(views)
    functions = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    urls = (WEBAPP / "dashboard" / "urls.py").read_text()
    for view in re.findall(r"views\.(\w+)", urls):
        assert view in functions, view
        decorators = {getattr(d, "id", getattr(d, "func", None) and getattr(d.func, "id", "")) for d in functions[view].decorator_list}
        assert "handles_errors" in decorators, f"{view} must turn database errors into a 503 page"


def test_no_django_orm_models():
    """Option B: data access goes through datatrust.storage only."""
    assert not (WEBAPP / "dashboard" / "models.py").exists()
    for py in WEBAPP.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "django.db" not in text, py
        assert "objects." not in text, py


def test_no_javascript_in_pages():
    for name in all_templates():
        assert "<script" not in template(name).lower(), name


# ------------------------------------------------------------------ with Django installed
def _manage(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(WEBAPP / "manage.py"), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_django_system_check():
    pytest.importorskip("django", reason="Django not installed: pip install -r requirements.txt")
    result = _manage("check", "--fail-level", "WARNING")
    assert result.returncode == 0, result.stdout + result.stderr


def test_django_page_tests():
    pytest.importorskip("django", reason="Django not installed: pip install -r requirements.txt")
    result = _manage("test", "dashboard", "--verbosity", "2")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

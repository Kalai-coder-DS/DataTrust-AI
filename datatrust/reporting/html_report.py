"""
datatrust/reporting/html_report.py

Renders a full analysis (the dict from pipeline.analyze_file) as ONE self-contained
HTML page: inline CSS, inline SVG, no JavaScript, no internet needed. It opens in any
browser, can be emailed, and prints cleanly.

Design choices:
  - Rendered from the analysis DICT, not live objects, so a stored analysis (MySQL,
    Phase 5b) can be re-rendered later without the original CSV.
  - Every value that came from the user's file is HTML-escaped (`_e`). A column named
    `<script>alert(1)</script>` must show up as text, not run as code (XSS).
  - Charts follow simple, readable forms: headline numbers as stat tiles; 0-100
    dimensions as single-hue bars; numeric columns as box plots; severity shown with
    an icon AND a label, never colour alone.
  - Light and dark mode via CSS custom properties.
"""
from __future__ import annotations

import html
from typing import Any

_SEVERITY = {  # icon + label + CSS class: severity is never shown by colour alone
    "critical": ("✖", "Critical", "sev-critical"),
    "high": ("▲", "High", "sev-high"),
    "medium": ("!", "Medium", "sev-medium"),
    "low": ("•", "Low", "sev-low"),
}
_GRADE_CLASS = {"A": "sev-good", "B": "sev-good", "C": "sev-medium", "D": "sev-high", "F": "sev-critical"}
_GRADE_MEANING = {"A": "excellent", "B": "good", "C": "usable with fixes", "D": "poor", "F": "not usable as-is"}

# Public names: the Django pages (Phase 6) show severities and grades exactly like the report.
SEVERITY_DISPLAY = _SEVERITY
GRADE_CLASS = _GRADE_CLASS
GRADE_MEANING = _GRADE_MEANING


def _e(value: Any) -> str:
    """Escape anything for safe inclusion in HTML."""
    return html.escape("" if value is None else str(value), quote=True)


def _num(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        a = abs(v)
        if a >= 1e7:
            return f"{v:.2e}"
        if a >= 100:
            return f"{v:,.0f}"
        return f"{v:,.2f}"
    return _e(v)


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


# ------------------------------------------------------------------ sections
def _header(a: dict) -> str:
    f, meta = a["file"], a["meta"]
    target = f"Target: <b>{_e(meta['target'])}</b>" if meta.get("target") else "No target column given"
    return f"""
<header>
  <p class="eyebrow">DataTrust AI · Data quality &amp; ML readiness report</p>
  <h1>{_e(f['file_name'])}</h1>
  <p class="sub">{f['rows']:,} rows × {f['columns']} columns · {_e(f['encoding'])} ·
     delimiter <code>{_e(f['delimiter'])}</code> · {target} ·
     analysed {_e(meta['analyzed_at'])}</p>
</header>"""


def _score_tile(r: dict) -> str:
    grade = r["grade"]
    bars = "".join(
        f"""<div class="dim" title="{_e(_label(d['name']))}: {d['score']} / 100 (weight {d['weight']:.0%})">
              <span class="dim-name">{_e(_label(d['name']))}</span>
              <span class="track"><span class="fill" style="width:{max(0.0, min(100.0, d['score']))}%"></span></span>
              <span class="dim-val">{d['score']:g}</span>
              <span class="dim-w">{d['weight']:.0%}</span>
            </div>"""
        for d in r["dimensions"]
    )
    caps = "".join(f'<p class="cap">⚠ {_e(c)}</p>' for c in r["caps"])
    return f"""
<section class="tile">
  <h2>{_e(r['name'])}</h2>
  <p class="hero"><span class="hero-num">{r['score']:.1f}</span><span class="hero-of">/ 100</span>
     <span class="badge {_GRADE_CLASS[grade]}">Grade {grade} · {_GRADE_MEANING[grade]}</span></p>
  <div class="dims" role="list">{bars}</div>
  {caps}
</section>"""


def _top_actions(a: dict) -> str:
    actions = a["scores"]["ml_readiness"]["top_actions"]
    if not actions:
        return '<section><h2>Top actions</h2><p class="muted">Nothing to fix. 🎉</p></section>'
    items = "".join(
        f"""<li><span class="gain">+{x['impact']:.1f}</span>
              <div><p>{_e(x['reason'])}</p><p class="muted">→ {_e(x['recommendation'])}</p></div></li>"""
        for x in actions
    )
    return f"""
<section><h2>Top actions <span class="muted small">(biggest score gain first)</span></h2>
  <ol class="actions">{items}</ol></section>"""


def _issues(a: dict) -> str:
    q = a["quality"]
    if not q["issues"]:
        return '<section><h2>Data quality issues</h2><p class="muted">No issues detected.</p></section>'
    rows = ""
    for i in q["issues"]:
        icon, label, cls = _SEVERITY[i["severity"]]
        examples = ", ".join(_e(x) for x in i["examples"][:3])
        rows += f"""<tr><td><span class="badge {cls}">{icon} {label}</span></td>
                    <td>{_e(i['column'] or '(whole dataset)')}</td>
                    <td>{_e(i['message'])}{f'<br><span class="muted">e.g. {examples}</span>' if examples else ''}</td>
                    <td class="num">{i['affected_rows']:,}</td>
                    <td>{_e(i['recommendation'])}</td></tr>"""
    c = q["severity_counts"]
    summary = " · ".join(f"{_SEVERITY[s][0]} {c[s]} {s}" for s in ("critical", "high", "medium", "low"))
    return f"""
<section><h2>Data quality issues <span class="muted small">{q['total_issues']} found · {summary}</span></h2>
<div class="scroll"><table><thead><tr><th>Severity</th><th>Column</th><th>Issue</th>
<th class="num">Rows</th><th>Recommendation</th></tr></thead><tbody>{rows}</tbody></table></div></section>"""


def _missing_chart(a: dict) -> str:
    cols = [c for c in a["profile"]["columns"] if c["null_pct"] > 0]
    if not cols:
        return ""
    cols.sort(key=lambda c: -c["null_pct"])
    peak = max(c["null_pct"] for c in cols)
    bars = "".join(
        f"""<div class="hbar" title="{_e(c['name'])}: {c['null_count']:,} missing ({c['null_pct']}%)">
              <span class="hbar-name">{_e(c['name'])}</span>
              <span class="hbar-track"><span class="fill" style="width:{c['null_pct'] / peak * 100:.1f}%"></span></span>
              <span class="hbar-val">{c['null_pct']}%</span></div>"""
        for c in cols
    )
    return f"""<section><h2>Missing values by column</h2>
  <p class="muted small">Blank cells only; placeholders such as 'Unknown' are listed under issues.</p>
  <div class="hbars">{bars}</div></section>"""


def _boxplot(s: dict) -> str:
    lo, hi = s["min"], s["max"]
    if hi <= lo:
        return '<span class="muted small">constant</span>'
    w, h, pad = 220, 26, 4

    def x(v: float) -> float:
        return pad + (v - lo) / (hi - lo) * (w - 2 * pad)

    tip = (f"min {_num(lo)} · Q1 {_num(s['q1'])} · median {_num(s['median'])} · "
           f"Q3 {_num(s['q3'])} · max {_num(hi)}")
    return f"""<svg class="box" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{tip}">
  <title>{tip}</title>
  <line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="13" y2="13" class="whisker"/>
  <line x1="{x(lo):.1f}" x2="{x(lo):.1f}" y1="8" y2="18" class="whisker"/>
  <line x1="{x(hi):.1f}" x2="{x(hi):.1f}" y1="8" y2="18" class="whisker"/>
  <rect x="{x(s['q1']):.1f}" y="5" width="{max(2.0, x(s['q3']) - x(s['q1'])):.1f}" height="16" rx="3" class="iqr"/>
  <line x1="{x(s['median']):.1f}" x2="{x(s['median']):.1f}" y1="4" y2="22" class="median"/>
</svg>"""


def _numeric(a: dict) -> str:
    nums = a["statistics"]["numeric"]
    if not nums:
        return ""
    rows = "".join(
        f"""<tr><td>{_e(s['column'])}</td><td class="num">{_num(s['mean'])}</td><td class="num">{_num(s['median'])}</td>
            <td class="num">{_num(s['std'])}</td><td>{_boxplot(s)}</td>
            <td>{_e(s['shape'])}{f'<br><span class="muted small">{s["excluded"]} invalid value(s) excluded</span>' if s['excluded'] else ''}</td></tr>"""
        for s in nums
    )
    return f"""<section><h2>Numeric columns</h2>
<p class="muted small">Box: middle 50% of values (Q1–Q3), line: median, whiskers: min–max. Each row has its own scale.</p>
<div class="scroll"><table><thead><tr><th>Column</th><th class="num">Mean</th><th class="num">Median</th>
<th class="num">Std</th><th>Distribution</th><th>Shape</th></tr></thead><tbody>{rows}</tbody></table></div></section>"""


def _categorical(a: dict) -> str:
    cats = a["statistics"]["categorical"]
    if not cats:
        return ""
    rows = "".join(
        f"""<tr><td>{_e(s['column'])}</td><td class="num">{s['n_categories']}</td>
            <td>{_e(s['top_value'])} <span class="muted">({s['top_pct']}%)</span></td>
            <td class="num">{s['rare_categories']}</td></tr>"""
        for s in cats
    )
    return f"""<section><h2>Label columns</h2><div class="scroll"><table><thead><tr><th>Column</th>
<th class="num">Categories</th><th>Most common</th><th class="num">Rare</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>"""


def _relationships(a: dict) -> str:
    assoc = a["statistics"]["associations"]
    if not assoc:
        return '<section><h2>Relationships between columns</h2><p class="muted">No strong relationships.</p></section>'
    rows = ""
    for x in assoc[:25]:
        if x["involves_target"]:
            tag = '<span class="badge sev-good">✔ Signal for target</span>'
        elif x["redundant"]:
            tag = '<span class="badge sev-medium">! Redundant features</span>'
        else:
            tag = ""
        pearson = x["details"].get("pearson")
        rows += f"""<tr><td>{_e(x['column_a'])} ~ {_e(x['column_b'])}</td><td>{_e(x['strength'])}</td>
                    <td>{_e(x['method'])} = {x['value']}{f' <span class="muted">(Pearson {pearson})</span>' if pearson is not None else ''}</td>
                    <td>{tag}</td></tr>"""
    more = f'<p class="muted small">… and {len(assoc) - 25} more in the JSON export.</p>' if len(assoc) > 25 else ""
    return f"""<section><h2>Relationships between columns</h2><div class="scroll"><table><thead><tr>
<th>Columns</th><th>Strength</th><th>Measure</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>{more}</section>"""


def _target(a: dict) -> str:
    t = a["statistics"]["target"]
    if not t:
        return ""
    if t["task"] == "regression":
        return (f"<section><h2>Target: {_e(t['column'])}</h2><p>Regression (predicting a number). "
                f"See its row under numeric columns.</p></section>")
    total = sum(t["class_counts"].values())
    bars = "".join(
        f"""<div class="hbar" title="{_e(k)}: {v:,} rows ({v / total:.1%})"><span class="hbar-name">{_e(k)}</span>
              <span class="hbar-track"><span class="fill" style="width:{v / total * 100:.1f}%"></span></span>
              <span class="hbar-val">{v / total:.1%}</span></div>"""
        for k, v in t["class_counts"].items()
    )
    return f"""<section><h2>Target: {_e(t['column'])}</h2>
<p>Classification, {t['n_classes']} classes · imbalance ratio {t['imbalance_ratio']} →
<b>{_e(t['balance'])}</b></p><div class="hbars">{bars}</div></section>"""


def _profile(a: dict) -> str:
    rows = "".join(
        f"""<tr><td>{_e(c['name'])}</td><td><code>{_e(c['pandas_dtype'])}</code></td><td>{_e(c['semantic_type'])}</td>
            <td class="num">{c['null_pct']}%</td><td class="num">{c['unique_count']:,}</td>
            <td class="muted">{_e(', '.join(str(v) for v in c['sample_values'][:3]))}</td></tr>"""
        for c in a["profile"]["columns"]
    )
    return f"""<section><h2>Column profile</h2><div class="scroll"><table><thead><tr><th>Column</th><th>Stored as</th>
<th>Meaning</th><th class="num">Missing</th><th class="num">Unique</th><th>Examples</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>"""


def _baseline(a: dict) -> str:
    """Phase 7: baseline model on raw vs cleaned data (only when the analysis includes one)."""
    from datatrust.ml.present import baseline_view
    v = baseline_view(a.get("ml"))
    if v is None:
        return ""
    title = '<h2>Baseline model <span class="muted small">does cleaning change what a model learns?</span></h2>'
    if v["status"] != "ok":
        return f'<section>{title}<p class="muted">Not trained: {_e(v["reason"])}</p></section>'
    bars = "".join(
        f"""<div class="dim mlbar" title="{_e(b['label'])}: {_e(b['text'])} ({_e(b['note'])})">
              <span class="dim-name">{_e(b['label'])}</span>
              <span class="track"><span class="fill" style="width:{b['width']}%"></span></span>
              <span class="dim-val">{_e(b['text'])}</span></div>"""
        for b in v["bars"])
    verdict = "".join(f"<p>{_e(t)}</p>" for t in v["verdict"])
    head = "".join(f'<th class="num">{_e(c)}</th>' for c in v["columns"])
    rows = "".join("<tr><td>" + _e(r["label"]) + "</td>" + "".join(f'<td class="num">{_e(c)}</td>' for c in r["cells"])
                   + "</tr>" for r in v["table"])
    steps = "".join(f"<li>{_e(x)}</li>" for x in v["steps"]) or "<li>Nothing to change.</li>"
    unused = "".join(f"<li><code>{_e(e['column'])}</code>: {_e(e['reason'])}</li>" for e in v["excluded"] + v["dropped"])
    notes = "".join(f'<p class="cap">{_e(n)}</p>' for n in v["notes"])
    change = f' · change after cleaning <b>{_e(v["change_text"])}</b>' if v["change_text"] else ""
    return f"""
<section>{title}
  <p class="muted">Predicting <b>{_e(v['target'])}</b> ({_e(v['task'])}). {_e(v['metric_label'])}: {_e(v['metric_about'])}{change}.</p>
  <div class="dims">{bars}</div>
  <div class="verdict">{verdict}</div>
  <div class="scroll"><table><thead><tr><th></th>{head}</tr></thead>
  <tbody>{rows}</tbody></table></div>
  <h3>What cleaning changed</h3><ul>{steps}</ul>
  {f'<h3>Columns not used</h3><ul>{unused}</ul>' if unused else ''}
  {notes}
  <p class="cap">{_e(v['footnote'])}</p>
</section>"""


def _notes(a: dict) -> str:
    notes = a["scores"]["trust"]["notes"] + a["scores"]["ml_readiness"]["notes"]
    warnings = a["file"]["warnings"]
    if not notes and not warnings:
        return ""
    items = "".join(f"<li>{_e(n)}</li>" for n in warnings + notes)
    return f"<section><h2>Notes</h2><ul>{items}</ul></section>"


_CSS = """
.viz-root{color-scheme:light;--surface-0:#f4f3f0;--surface-1:#fcfcfb;--border:#e2e1dc;
 --text-primary:#0b0b0b;--text-secondary:#52514e;--series-1:#2a78d6;--track:#ebeae6;
 --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b}
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;
 --border:#33332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;--series-1:#3987e5;--track:#2a2a27}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
 font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 20px 64px}
header{margin-bottom:24px}.eyebrow{color:var(--text-secondary);font-size:13px;margin:0;letter-spacing:.02em}
h1{margin:4px 0;font-size:28px}h2{font-size:18px;margin:0 0 12px}.sub{color:var(--text-secondary);margin:0}
section{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:20px;margin:16px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.tiles section{margin:0}
.hero{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin:0 0 16px}
.hero-num{font-size:48px;font-weight:700;line-height:1}.hero-of{color:var(--text-secondary)}
.dim{display:grid;grid-template-columns:120px 1fr 36px 36px;align-items:center;gap:10px;margin:6px 0}
.dim-name{color:var(--text-secondary);font-size:14px}.dim-val{text-align:right;font-variant-numeric:tabular-nums}
.dim-w{color:var(--text-secondary);font-size:12px;text-align:right}
.track,.hbar-track{background:var(--track);border-radius:4px;height:10px;overflow:hidden;display:block}
.fill{display:block;height:100%;background:var(--series-1);border-radius:0 4px 4px 0}
.hbar{display:grid;grid-template-columns:minmax(80px,180px) 1fr 64px;gap:10px;align-items:center;margin:6px 0}
.hbar-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.hbar-val{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:13px;font-weight:600;white-space:nowrap;
 border:1px solid currentColor;color:var(--text-primary)}
.sev-critical{border-color:var(--critical);background:color-mix(in srgb,var(--critical) 16%,transparent)}
.sev-high{border-color:var(--serious);background:color-mix(in srgb,var(--serious) 18%,transparent)}
.sev-medium{border-color:var(--warning);background:color-mix(in srgb,var(--warning) 20%,transparent)}
.sev-low{border-color:var(--border)}
.sev-good{border-color:var(--good);background:color-mix(in srgb,var(--good) 14%,transparent)}
.cap{margin:12px 0 0;color:var(--text-secondary);font-size:14px}
.actions{margin:0;padding:0;list-style:none}.actions li{display:flex;gap:14px;padding:10px 0;border-top:1px solid var(--border)}
.actions li:first-child{border-top:0}.actions p{margin:0}
.gain{font-weight:700;font-variant-numeric:tabular-nums;min-width:52px;color:var(--text-primary)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;color:var(--text-secondary);font-weight:600;border-bottom:1px solid var(--border);padding:8px}
td{border-bottom:1px solid var(--border);padding:8px;vertical-align:top}.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:var(--text-secondary)}.small{font-size:13px;font-weight:400}
code{font-size:13px}
h3{font-size:15px;margin:16px 0 6px}.mlbar{grid-template-columns:120px 1fr 64px}.verdict p{margin:10px 0}
.box .whisker{stroke:var(--text-secondary);stroke-width:1.5}.box .iqr{fill:var(--series-1);opacity:.35;stroke:var(--series-1)}
.box .median{stroke:var(--text-primary);stroke-width:2}
footer{color:var(--text-secondary);font-size:13px;margin-top:24px}
@media print{body{background:#fff}section{break-inside:avoid}}
"""


def render_html(analysis: dict) -> str:
    """Render a full analysis dict (from pipeline.analyze_file) as a self-contained HTML page."""
    s = analysis["scores"]
    body = (
        _header(analysis)
        + f'<div class="tiles">{_score_tile(s["trust"])}{_score_tile(s["ml_readiness"])}</div>'
        + _top_actions(analysis)
        + _issues(analysis)
        + _missing_chart(analysis)
        + _target(analysis)
        + _numeric(analysis)
        + _categorical(analysis)
        + _relationships(analysis)
        + _baseline(analysis)
        + _profile(analysis)
        + _notes(analysis)
    )
    title = f"DataTrust AI report – {analysis['file']['file_name']}"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{_CSS}</style></head>
<body class="viz-root"><main>{body}
<footer>Generated by DataTrust AI v{_e(analysis['meta']['version'])}. Every point lost in a score is listed
as a deduction; see docs/ARCHITECTURE.md for the scoring rules.</footer></main></body></html>"""

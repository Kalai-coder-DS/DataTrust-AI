"""
datatrust/web/charts.py

The score-history chart on the dataset page, as inline SVG (no JavaScript, no chart
library), in the same style as the HTML report.

Design (see the dataviz rules used for the report):
  - ONE y-axis, 0-100, shared by both scores (they are on the same scale).
  - Two series in fixed categorical slots: Trust = slot 1 (blue), ML Readiness = slot 2
    (orange). Palette validated for colour-blind separation in light and dark mode.
  - Identity never by colour alone: ML Readiness also has a dashed line and square markers,
    there is a legend, and both lines are direct-labelled at the latest run.
  - Hover: every marker has a <title>, so the browser shows a native tooltip.
  - The runs table under the chart is the accessible table view of the same numbers.
  - Every piece of text is HTML-escaped (dataset names come from uploaded file names).
"""
from __future__ import annotations

import html

WIDTH, HEIGHT = 720, 240
LEFT, RIGHT, TOP, BOTTOM = 40, 120, 16, 36   # right margin holds the direct labels
SERIES = (
    # key in history row, label, CSS variable, dashed?, marker shape
    ("trust_score", "Trust", "--series-1", False, "circle"),
    ("ml_score", "ML Readiness", "--series-2", True, "square"),
)


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _x(i: int, n: int) -> float:
    plot_w = WIDTH - LEFT - RIGHT
    return LEFT + (plot_w / 2 if n == 1 else plot_w * i / (n - 1))


def _y(score: float) -> float:
    plot_h = HEIGHT - TOP - BOTTOM
    return TOP + plot_h * (1 - max(0.0, min(100.0, float(score))) / 100)


def _marker(shape: str, x: float, y: float, color: str, tip: str) -> str:
    # ML Readiness is capped at Trust, so the two often sit on the SAME point. Trust is a
    # hollow ring drawn on top of a smaller filled ML square: both stay visible when equal.
    title = f"<title>{_e(tip)}</title>"
    if shape == "square":
        return (f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5" '
                f'fill="var({color})" stroke="var(--surface-1)" stroke-width="1.5">{title}</rect>')
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.5" fill="none" stroke="var({color})" '
            f'stroke-width="2.5">{title}</circle>')


def trend_svg(history: list[dict], dataset_name: str = "") -> str:
    """
    Line chart of Trust and ML Readiness for every run of one dataset (oldest first,
    the order AnalysisRepository.history returns). Returns '' when there are no runs.
    """
    if not history:
        return ""
    n = len(history)
    parts: list[str] = []

    # Recessive grid + y labels (one axis only).
    for v in (0, 50, 100):
        y = _y(v)
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="axis" x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{v}</text>')

    # X labels: run numbers (all of them when few, otherwise first and last only).
    label_idx = range(n) if n <= 12 else (0, n - 1)
    for i in label_idx:
        parts.append(f'<text class="axis" x="{_x(i, n):.1f}" y="{HEIGHT - 12}" '
                     f'text-anchor="middle">#{_e(history[i]["run_id"])}</text>')

    # Lines first (Trust solid, then ML dashed on top so blue shows through the gaps when
    # they coincide), then markers: ML squares, with the hollow Trust rings on top.
    end_labels, markers = [], []
    for key, label, color, dashed, shape in SERIES:
        pts = [(_x(i, n), _y(row[key])) for i, row in enumerate(history)]
        if n > 1:
            dash = ' stroke-dasharray="6 4"' if dashed else ""
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{path}" fill="none" stroke="var({color})" stroke-width="2" '
                         f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        series_markers = []
        for (x, y), row in zip(pts, history):
            tip = (f"Run #{row['run_id']} · {row['analyzed_at']} UTC\n"
                   f"{label}: {float(row[key]):.1f} (grade {row[key.replace('score', 'grade')]})")
            series_markers.append(_marker(shape, x, y, color, tip))
        markers.insert(0, series_markers)             # later series drawn first
        end_labels.append([pts[-1][1], f"{label} {float(history[-1][key]):.1f}"])
    for series_markers in markers:
        parts.extend(series_markers)

    # Direct labels at the latest run; nudge apart if they would overlap.
    end_labels.sort(key=lambda t: t[0])
    if len(end_labels) == 2 and end_labels[1][0] - end_labels[0][0] < 14:
        mid = (end_labels[0][0] + end_labels[1][0]) / 2
        end_labels[0][0], end_labels[1][0] = mid - 7, mid + 7
    x_end = _x(n - 1, n) + 10
    for y, text in end_labels:
        parts.append(f'<text class="direct" x="{x_end:.1f}" y="{y + 4:.1f}">{_e(text)}</text>')

    title = f"Score history for {dataset_name}" if dataset_name else "Score history"
    desc = f"{n} run(s). Latest: " + ", ".join(t for _, t in sorted(end_labels, key=lambda t: t[1]))
    return (f'<svg class="trend" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
            f'aria-labelledby="trend-title trend-desc" preserveAspectRatio="xMidYMid meet">'
            f'<title id="trend-title">{_e(title)}</title><desc id="trend-desc">{_e(desc)}</desc>'
            + "".join(parts) + "</svg>")

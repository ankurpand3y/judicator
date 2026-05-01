from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from judicator.biases.base import BiasResult
from judicator.fixtures import FIXTURE_VERSION

# ── box layout ─────────────────────────────────────────────────────────────────
# Total width = 64 chars (including ║ borders).
# Table column widths: [17, 8, 8, 10, 15] → 58 inner + 4 ║ inner + 2 ║ outer = 64

_W = 64
_INNER = _W - 2           # 62
_COL = [20, 7, 7, 10, 14]  # TEST, SCORE, RANK, VERDICT, SEVERITY

_ATTRIBUTION = (
    "Built on: OffsetBias, JudgeBench, MT-Bench, BeaverTails, SummEval, DSTC11"
)


def _sep(left: str = "╠", right: str = "╣", fill: str = "═") -> str:
    return left + fill * _INNER + right


def _tsep(left: str, mid: str, right: str) -> str:
    return left + mid.join(fill * w for fill, w in zip(["═"] * 5, _COL)) + right


def _line(text: str = "") -> str:
    return f"║  {text:<{_INNER - 2}}║"


def _trow(*cells: str) -> str:
    parts = [f" {c[:w - 1]:<{w - 1}}" for c, w in zip(cells, _COL)]
    return "║" + "║".join(parts) + "║"


def _wrap(text: str, width: int = _INNER - 4) -> list[str]:
    """Word-wrap text to fit inside the box."""
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 <= width:
            line = f"{line} {word}".lstrip()
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


# ── report ─────────────────────────────────────────────────────────────────────

@dataclass
class AuditReport:
    judge_name: str
    judge_type: str   # lowercase: "pointwise" | "pairwise" | "binary"
    domain: str
    timestamp: str    # ISO 8601 UTC
    tests: dict[str, BiasResult]

    # ── convenience ────────────────────────────────────────────────────────────

    def failed_tests(self) -> list[BiasResult]:
        return [r for r in self.tests.values() if r.verdict == "FAIL"]

    def passed_tests(self) -> list[BiasResult]:
        return [r for r in self.tests.values() if r.verdict == "PASS"]

    def ranked(self) -> list[BiasResult]:
        """Return applicable results sorted worst-first (score ascending)."""
        return sorted(
            [r for r in self.tests.values() if not r.not_applicable],
            key=lambda r: r.score,
        )

    def summary(self) -> str:
        """Return full console report as a string (also printed by audit())."""
        return _render_console(self)

    # ── export ─────────────────────────────────────────────────────────────────

    def save_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(_to_dict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_html(self, path: str) -> None:
        Path(path).write_text(_render_html(self), encoding="utf-8")


# ── console renderer ───────────────────────────────────────────────────────────

def _render_console(r: AuditReport) -> str:
    n_applicable = len(r.ranked())
    n_fail = len(r.failed_tests())
    n_pass = len(r.passed_tests())
    na_tests = [t for t in r.tests.values() if t.not_applicable]

    lines: list[str] = []

    # Header
    lines += [
        _sep("╔", "╗"),
        _line("JUDICATOR — AUDIT REPORT"),
        _sep(),
        _line(f"Judge:   {r.judge_name}"),
        _line(f"Domain:  {r.domain}"),
        _line(f"Type:    {r.judge_type}"),
        _line(f"Tested:  {r.timestamp}"),
    ]

    # Table header
    lines += [
        _tsep("╠", "╦", "╣"),
        _trow("  BIAS TEST", " SCORE", " RANK", " VERDICT", " SEVERITY"),
        _tsep("╠", "╬", "╣"),
    ]

    # Applicable results (worst → best)
    for res in r.ranked():
        lines.append(_trow(
            f"  {res.test_name}",
            f" {res.score:.3f}",
            f" {res.rank}/{n_applicable}",
            f" {res.verdict}",
            f" {res.severity or ''}",
        ))

    # N/A results
    if na_tests:
        lines.append(_tsep("╠", "╬", "╣"))
        for res in na_tests:
            # severity col shows the judge type the test requires (first word of skip reason)
            hint = res.skip_reason.split()[0] if res.skip_reason else "—"
            lines.append(_trow(
                f"  {res.test_name}",
                "  —",
                "  —",
                " N/A",
                f" ({hint} only)",
            ))

    # Top findings
    fails = r.failed_tests()
    if fails:
        lines.append(_sep())
        lines.append(_line("TOP FINDINGS"))
        lines.append(_line())
        for i, res in enumerate(sorted(fails, key=lambda x: x.score)[:3], 1):
            sev = res.severity or ""
            lines.append(_line(f"  {i} [{sev}] {res.test_name} — score {res.score:.3f}"))
            # Show first example if available
            if res.examples:
                ex = res.examples[0]
                for k, v in ex.items():
                    vstr = str(v)[:50]
                    lines.append(_line(f"    {k}: {vstr}"))
            # Show key detail
            for k, v in list(res.details.items())[:2]:
                if k != "error":
                    lines.append(_line(f"    {k}: {v}"))
            lines.append(_line())

    # Footer
    lines += [
        _sep(),
        _line(f"  {n_fail} tests FAILED · {n_pass} tests PASSED · {len(na_tests)} tests N/A"),
        _line(),
    ]
    for chunk in _wrap(_ATTRIBUTION):
        lines.append(_line(f"  {chunk}"))
    lines.append(_sep("╚", "╝"))

    return "\n".join(lines)


# ── JSON serialiser ────────────────────────────────────────────────────────────

def _to_dict(r: AuditReport) -> dict:
    ranked = r.ranked()
    fails = r.failed_tests()
    worst = min(ranked, key=lambda x: x.score) if ranked else None

    tests_out: dict = {}
    for name, res in r.tests.items():
        if res.not_applicable:
            tests_out[name] = {
                "not_applicable": True,
                "skip_reason": res.skip_reason,
            }
        else:
            tests_out[name] = {
                "score": res.score,
                "rank": res.rank,
                "verdict": res.verdict,
                "severity": res.severity,
                "n_fixtures": res.n_fixtures,
                "not_applicable": False,
                "details": res.details,
                "examples": res.examples,
            }

    return {
        "judicator_version": "0.2.0",
        "timestamp": r.timestamp,
        "judge": {
            "name": r.judge_name,
            "type": r.judge_type,
            "domain": r.domain,
        },
        "summary": {
            "tests_run": len(ranked),
            "tests_failed": len(fails),
            "tests_passed": len(r.passed_tests()),
            "tests_na": len([t for t in r.tests.values() if t.not_applicable]),
            "worst_bias": worst.test_name if worst else None,
            "worst_bias_score": worst.score if worst else None,
        },
        "tests": tests_out,
        "fixtures": {
            "source": "pre-shipped-v0.1",
            "fixture_version": FIXTURE_VERSION,
            "domain": r.domain,
        },
    }


# ── HTML renderer ──────────────────────────────────────────────────────────────

_VERDICT_COLOR = {"PASS": "#2e7d32", "FAIL": "#c62828", "N/A": "#757575"}
_SEV_COLOR = {
    "CRITICAL": "#c62828", "SIGNIFICANT": "#e65100",
    "MINOR": "#f57c00", "NONE": "#2e7d32",
}


def _render_html(r: AuditReport) -> str:
    n_fail = len(r.failed_tests())
    n_pass = len(r.passed_tests())
    na_tests = [t for t in r.tests.values() if t.not_applicable]
    n_applicable = len(r.ranked())

    def badge(text: str, color: str) -> str:
        return (
            f'<span style="color:{color};font-weight:bold;'
            f'padding:1px 6px;border-radius:3px;'
            f'border:1px solid {color}">{text}</span>'
        )

    rows = ""
    for res in r.ranked():
        vc = _VERDICT_COLOR.get(res.verdict, "#000")
        sc = _SEV_COLOR.get(res.severity or "", "#000")
        rows += (
            f"<tr>"
            f"<td>{res.test_name}</td>"
            f"<td style='text-align:center'>{res.score:.3f}</td>"
            f"<td style='text-align:center'>{res.rank}/{n_applicable}</td>"
            f"<td style='text-align:center'>{badge(res.verdict, vc)}</td>"
            f"<td style='text-align:center'>{badge(res.severity or '—', sc)}</td>"
            f"</tr>\n"
        )
    for res in na_tests:
        rows += (
            f"<tr style='color:#999'>"
            f"<td>{res.test_name}</td>"
            f"<td colspan='4' style='text-align:center'>N/A — {res.skip_reason}</td>"
            f"</tr>\n"
        )

    findings_html = ""
    fails = r.failed_tests()
    if fails:
        findings_html = "<h2>Top Findings</h2>"
        for res in sorted(fails, key=lambda x: x.score)[:3]:
            sc = _SEV_COLOR.get(res.severity or "", "#000")
            findings_html += (
                f"<div style='margin:12px 0;padding:10px;border-left:4px solid {sc}'>"
                f"<b>[{res.severity}] {res.test_name}</b> — score {res.score:.3f}<br>"
            )
            for k, v in list(res.details.items())[:3]:
                findings_html += f"<small>{k}: {v}</small><br>"
            if res.examples:
                ex = res.examples[0]
                for k, v in list(ex.items())[:3]:
                    findings_html += f"<small>{k}: {str(v)[:80]}</small><br>"
            findings_html += "</div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Judicator Audit — {r.judge_name}</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#222}}
  h1{{color:#1a237e;border-bottom:2px solid #1a237e;padding-bottom:8px}}
  table{{border-collapse:collapse;width:100%;margin:16px 0}}
  th{{background:#1a237e;color:#fff;padding:8px 12px;text-align:left}}
  td{{padding:8px 12px;border-bottom:1px solid #e0e0e0}}
  tr:hover td{{background:#f5f5f5}}
  .meta{{color:#555;margin-bottom:16px}}
  .footer{{margin-top:24px;padding-top:12px;border-top:1px solid #e0e0e0;font-size:0.85em;color:#777}}
  .summary-bar{{display:flex;gap:16px;margin:16px 0}}
  .stat{{padding:8px 16px;border-radius:4px;font-weight:bold}}
  .stat-fail{{background:#ffebee;color:#c62828}}
  .stat-pass{{background:#e8f5e9;color:#2e7d32}}
  .stat-na{{background:#f5f5f5;color:#555}}
</style>
</head>
<body>
<h1>Judicator — Audit Report</h1>
<div class="meta">
  <b>Judge:</b> {r.judge_name} &nbsp;|&nbsp;
  <b>Type:</b> {r.judge_type} &nbsp;|&nbsp;
  <b>Domain:</b> {r.domain} &nbsp;|&nbsp;
  <b>Tested:</b> {r.timestamp}
</div>
<div class="summary-bar">
  <div class="stat stat-fail">{n_fail} FAILED</div>
  <div class="stat stat-pass">{n_pass} PASSED</div>
  <div class="stat stat-na">{len(na_tests)} N/A</div>
</div>
<table>
<thead><tr>
  <th>Bias Test</th><th>Score</th><th>Rank</th><th>Verdict</th><th>Severity</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
{findings_html}
<div class="footer">
  Judicator v0.2.0 &nbsp;·&nbsp; {_ATTRIBUTION}
</div>
</body>
</html>"""

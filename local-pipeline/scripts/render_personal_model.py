#!/usr/bin/env python3
"""SelfTrace Stage 7 — Evidence Presentation (v2.3).

Consumes Stage-6 personal_model.json and renders a self-contained local HTML
report. No LLM/API call is made in this stage.

Pipeline:
    personal_model.json
        -> presentation view model
        -> self-contained HTML
        -> clickable evidence provenance

The generated page is portable: CSS and JavaScript are embedded in the HTML.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_VERSION = "2.3"


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def badge(text: str, kind: str = "") -> str:
    cls = f"badge {kind}".strip()
    return f'<span class="{cls}">{esc(text)}</span>'


def unique_post_sources(evidence_trace: list[dict]) -> list[dict]:
    """Flatten nested Stage-6 provenance to unique posts."""
    posts: dict[str, dict] = {}

    for trace in evidence_trace or []:
        trace_label = (
            trace.get("canonical_signal")
            or trace.get("canonical_topic")
            or trace.get("theme")
            or trace.get("ref_id")
            or "Evidence"
        )

        for item in trace.get("evidence") or []:
            post_id = str(item.get("post_id") or "")
            if not post_id:
                continue

            entry = posts.setdefault(post_id, {
                "post_id": post_id,
                "created_at": item.get("created_at"),
                "source_url": item.get("source_url"),
                "claims": [],
                "evidence": [],
                "topics": [],
                "trace_labels": [],
            })

            claim = item.get("claim")
            if claim and claim not in entry["claims"]:
                entry["claims"].append(claim)

            for ev in item.get("evidence") or []:
                if ev and ev not in entry["evidence"]:
                    entry["evidence"].append(ev)

            for topic in item.get("primary_topics") or []:
                if topic and topic not in entry["topics"]:
                    entry["topics"].append(topic)

            if trace_label and trace_label not in entry["trace_labels"]:
                entry["trace_labels"].append(trace_label)

        # Topic trajectories may preserve only post ids + evidence metadata.
        for post_id in trace.get("post_ids") or []:
            post_id = str(post_id)
            posts.setdefault(post_id, {
                "post_id": post_id,
                "created_at": None,
                "source_url": None,
                "claims": [],
                "evidence": [],
                "topics": [],
                "trace_labels": [trace_label] if trace_label else [],
            })

    return sorted(
        posts.values(),
        key=lambda x: (x.get("created_at") or "", x["post_id"]),
    )


def render_evidence(evidence_trace: list[dict], evidence_ref_ids: list[str]) -> str:
    posts = unique_post_sources(evidence_trace)
    refs = " ".join(badge(x, "ref") for x in (evidence_ref_ids or []))

    if not posts:
        return f"""
        <details class="evidence">
          <summary>Evidence trace {refs}</summary>
          <p class="muted">No expanded post-level provenance is available in this record.</p>
        </details>
        """

    post_html = []
    for p in posts:
        url = p.get("source_url")
        title = f"Post {esc(p['post_id'])}"
        if url:
            title = f'<a href="{esc(url)}" target="_blank" rel="noopener">{title}</a>'

        date = esc(p.get("created_at") or "date unavailable")
        trace_labels = " · ".join(esc(x) for x in p.get("trace_labels") or [])
        topics = ", ".join(esc(x) for x in p.get("topics") or [])

        detail_parts = [f"<div class='evidence-post-head'>{title}<span>{date}</span></div>"]
        if trace_labels:
            detail_parts.append(f"<div class='trace-label'>{trace_labels}</div>")
        if topics:
            detail_parts.append(f"<div class='small'><strong>Topics:</strong> {topics}</div>")

        for claim in p.get("claims") or []:
            detail_parts.append(f"<div class='source-claim'>{esc(claim)}</div>")

        if p.get("evidence"):
            ev_items = "".join(f"<li>{esc(x)}</li>" for x in p["evidence"])
            detail_parts.append(f"<ul class='source-evidence'>{ev_items}</ul>")

        post_html.append(f"<div class='evidence-post'>{''.join(detail_parts)}</div>")

    return f"""
    <details class="evidence">
      <summary>Evidence trace · {len(posts)} supporting post(s) {refs}</summary>
      <div class="evidence-body">
        {''.join(post_html)}
      </div>
    </details>
    """


def render_claim(claim: dict) -> str:
    return f"""
    <article class="claim-card" id="{esc(claim.get('claim_id'))}">
      <div class="claim-top">
        <span class="claim-id">{esc(claim.get('claim_id'))}</span>
        <div class="badges">
          {badge(claim.get('dimension', ''), 'dimension')}
          {badge(claim.get('scope', ''), 'scope')}
          {badge(claim.get('confidence', ''), 'confidence')}
        </div>
      </div>
      <h3>{esc(claim.get('statement'))}</h3>
      <p class="rationale">{esc(claim.get('rationale'))}</p>
      {render_evidence(claim.get('evidence_trace') or [], claim.get('evidence_ref_ids') or [])}
    </article>
    """


def render_current_state(item: dict) -> str:
    return f"""
    <article class="state-card">
      <div class="badges">
        {badge(item.get('kind', ''), 'scope')}
        {badge(item.get('confidence', ''), 'confidence')}
      </div>
      <h3>{esc(item.get('label'))}</h3>
      <p>{esc(item.get('rationale'))}</p>
      {render_evidence(item.get('evidence_trace') or [], item.get('evidence_ref_ids') or [])}
    </article>
    """


def render_evolution(item: dict) -> str:
    period = " → ".join(
        str(x) for x in [item.get("start_period"), item.get("end_period")] if x
    )
    period_html = f"<div class='period'>{esc(period)}</div>" if period else ""
    return f"""
    <article class="evolution-card">
      <div class="badges">{badge(item.get('confidence', ''), 'confidence')}</div>
      {period_html}
      <h3>{esc(item.get('statement'))}</h3>
      <p>{esc(item.get('rationale'))}</p>
      {render_evidence(item.get('evidence_trace') or [], item.get('evidence_ref_ids') or [])}
    </article>
    """


def build_html(stage6: dict) -> str:
    model = stage6.get("personal_model") or {}
    claims = model.get("claims") or []
    domain_index = model.get("domain_index") or {}

    claim_by_id = {
        c.get("claim_id"): c for c in claims if c.get("claim_id")
    }

    domain_meta = [
        ("COGNITIVE", "Cognitive Model", "Knowledge and recurring reasoning operations"),
        ("MOTIVATIONAL", "Motivational Model", "Interests, aims and goal orientation"),
        ("BEHAVIORAL", "Behavioral Model", "Observed activities and practical engagement"),
        ("COMMUNICATIVE", "Communicative Model", "Recurring ways of expressing and structuring ideas"),
    ]

    sections = []
    for key, title, subtitle in domain_meta:
        ids = domain_index.get(key) or []
        domain_claims = [claim_by_id[x] for x in ids if x in claim_by_id]
        if not domain_claims:
            continue
        sections.append(f"""
        <section id="{key.lower()}">
          <div class="section-head">
            <div>
              <div class="eyebrow">{esc(key)}</div>
              <h2>{esc(title)}</h2>
              <p>{esc(subtitle)}</p>
            </div>
            <span class="count">{len(domain_claims)} claims</span>
          </div>
          <div class="cards">{''.join(render_claim(c) for c in domain_claims)}</div>
        </section>
        """)

    current_state = model.get("current_state") or []
    evolution = model.get("evolution_patterns") or []
    limitations = model.get("limitations") or []

    source_dataset = ((stage6.get("source_stage5") or {}).get("dataset") or {})
    dataset_bits = [
        ("Posts", source_dataset.get("post_count")),
        ("Signals", source_dataset.get("signal_count")),
        ("From", source_dataset.get("first_seen")),
        ("To", source_dataset.get("last_seen")),
    ]
    stats = "".join(
        f"<div class='stat'><span>{esc(k)}</span><strong>{esc(v)}</strong></div>"
        for k, v in dataset_bits if v is not None
    )

    current_html = ""
    if current_state:
        current_html = f"""
        <section id="current">
          <div class="section-head">
            <div>
              <div class="eyebrow">NOW</div>
              <h2>Current / Emerging State</h2>
              <p>What the available traces most strongly indicate in the recent period.</p>
            </div>
            <span class="count">{len(current_state)} items</span>
          </div>
          <div class="cards state-grid">
            {''.join(render_current_state(x) for x in current_state)}
          </div>
        </section>
        """

    evolution_html = ""
    if evolution:
        evolution_html = f"""
        <section id="evolution">
          <div class="section-head">
            <div>
              <div class="eyebrow">TIME</div>
              <h2>Evolution Patterns</h2>
              <p>Observed changes and continuities across the available trace window.</p>
            </div>
            <span class="count">{len(evolution)} patterns</span>
          </div>
          <div class="cards">
            {''.join(render_evolution(x) for x in evolution)}
          </div>
        </section>
        """

    limit_html = "".join(f"<li>{esc(x)}</li>" for x in limitations)

    nav_links = [
        ("overview", "Overview"),
        ("cognitive", "Cognitive"),
        ("motivational", "Motivational"),
        ("behavioral", "Behavioral"),
        ("communicative", "Communicative"),
        ("current", "Current"),
        ("evolution", "Evolution"),
        ("boundaries", "Boundaries"),
    ]
    nav = "".join(f'<a href="#{a}">{b}</a>' for a, b in nav_links)

    generated = stage6.get("generated_at") or ""
    provider = stage6.get("provider") or ""
    model_name = stage6.get("model") or ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SelfTrace Personal Model</title>
<style>
:root {{
  --bg: #f5f5f2;
  --panel: #ffffff;
  --ink: #20211f;
  --muted: #6b6c68;
  --line: #deded8;
  --soft: #efefe9;
  --accent: #4d5b50;
  --shadow: 0 7px 24px rgba(0,0,0,.045);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
  line-height: 1.6;
}}
a {{ color: inherit; }}
.shell {{ max-width: 1120px; margin: 0 auto; padding: 0 28px 80px; }}
nav {{
  position: sticky; top: 0; z-index: 20;
  display: flex; flex-wrap: wrap; gap: 8px;
  padding: 14px 0;
  background: color-mix(in srgb, var(--bg) 91%, transparent);
  backdrop-filter: blur(10px);
}}
nav a {{
  text-decoration: none; font-size: 13px; padding: 6px 10px;
  border: 1px solid var(--line); border-radius: 999px; background: var(--panel);
}}
.hero {{
  padding: 76px 0 42px;
  border-bottom: 1px solid var(--line);
}}
.kicker, .eyebrow {{
  text-transform: uppercase; letter-spacing: .13em; font-size: 12px;
  color: var(--muted); font-weight: 700;
}}
.hero h1 {{ font-size: clamp(42px, 7vw, 76px); line-height: 1.03; margin: 12px 0 20px; letter-spacing: -.045em; }}
.hero-summary {{ max-width: 900px; font-size: 20px; color: #393a37; }}
.meta {{ color: var(--muted); font-size: 13px; margin-top: 22px; }}
.stats {{
  display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
  gap: 10px; margin-top: 28px;
}}
.stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; }}
.stat span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.stat strong {{ font-size: 15px; font-weight: 600; }}
section {{ padding: 54px 0 0; }}
.section-head {{
  display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
  margin-bottom:18px;
}}
.section-head h2 {{ font-size: 31px; line-height:1.15; letter-spacing:-.025em; margin:5px 0 4px; }}
.section-head p {{ margin:0; color:var(--muted); }}
.count {{ color:var(--muted); font-size:13px; white-space:nowrap; }}
.cards {{ display:grid; gap:14px; }}
.state-grid {{ grid-template-columns: repeat(auto-fit,minmax(310px,1fr)); }}
.claim-card, .state-card, .evolution-card {{
  background:var(--panel); border:1px solid var(--line); border-radius:18px;
  padding:20px 22px; box-shadow:var(--shadow);
}}
.claim-top {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }}
.claim-id {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color:var(--muted); font-size:13px; }}
.badges {{ display:flex; flex-wrap:wrap; gap:6px; }}
.badge {{
  display:inline-flex; align-items:center; padding:3px 8px; border-radius:999px;
  background:var(--soft); border:1px solid var(--line); font-size:11px; letter-spacing:.025em;
}}
.badge.confidence {{ font-weight:700; }}
.badge.ref {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.claim-card h3, .state-card h3, .evolution-card h3 {{
  font-size:19px; line-height:1.45; margin:13px 0 7px; letter-spacing:-.012em;
}}
.rationale, .state-card p, .evolution-card p {{ color:#51524e; margin:0 0 10px; }}
.period {{ margin-top:10px; color:var(--muted); font-size:13px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.evidence {{ margin-top:16px; border-top:1px solid var(--line); padding-top:12px; }}
.evidence summary {{ cursor:pointer; color:#484a46; font-size:13px; }}
.evidence-body {{ display:grid; gap:9px; margin-top:12px; }}
.evidence-post {{ background:#fafaf7; border:1px solid var(--line); border-radius:12px; padding:12px 14px; }}
.evidence-post-head {{ display:flex; justify-content:space-between; gap:12px; font-size:13px; }}
.evidence-post-head span {{ color:var(--muted); font-size:12px; }}
.trace-label {{ margin-top:5px; color:var(--accent); font-weight:600; font-size:12px; }}
.small {{ font-size:12px; color:var(--muted); margin-top:5px; }}
.source-claim {{ margin-top:9px; font-size:13px; }}
.source-evidence {{ margin:7px 0 0 18px; padding:0; color:#5c5d59; font-size:12px; }}
.muted {{ color:var(--muted); }}
.boundaries {{
  background:#ecece6; border:1px solid var(--line); border-radius:18px; padding:22px 24px;
}}
.boundaries ul {{ margin:10px 0 0; padding-left:20px; }}
.footer {{ margin-top:50px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }}
@media(max-width:700px) {{
  .shell {{ padding-left:18px; padding-right:18px; }}
  .hero {{ padding-top:48px; }}
  .section-head {{ align-items:flex-start; flex-direction:column; gap:4px; }}
  .claim-top {{ align-items:flex-start; flex-direction:column; }}
  .evidence-post-head {{ flex-direction:column; gap:2px; }}
}}
</style>
</head>
<body>
<div class="shell">
  <nav>{nav}</nav>

  <header class="hero" id="overview">
    <div class="kicker">SelfTrace · Evidence-based longitudinal personal model</div>
    <h1>Personal Model</h1>
    <div class="hero-summary">{esc(model.get("overall_summary") or "")}</div>
    <div class="stats">{stats}</div>
    <div class="meta">Generated {esc(generated)} · {esc(provider)} / {esc(model_name)} · Stage 7 renderer v{STAGE_VERSION}</div>
  </header>

  {''.join(sections)}
  {current_html}
  {evolution_html}

  <section id="boundaries">
    <div class="section-head">
      <div>
        <div class="eyebrow">BOUNDARY</div>
        <h2>Evidence Boundaries</h2>
        <p>What this model should and should not be interpreted as.</p>
      </div>
    </div>
    <div class="boundaries"><ul>{limit_html}</ul></div>
  </section>

  <div class="footer">
    SelfTrace · Digital Trace → Evidence → Temporal Pattern → Personal Model → Evidence Presentation
  </div>
</div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SelfTrace Stage 7 — Evidence Presentation v2.3")
    p.add_argument("--input", default="data/personal/personal_model.json", type=Path)
    p.add_argument("--output-dir", default="data/presentation", type=Path)
    p.add_argument("--filename", default="personal_model.html")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage6 = load_json(input_path)
    schema = str(stage6.get("schema_version") or "")
    if not schema.startswith("selftrace-personal-model-v2"):
        raise ValueError(f"Unsupported Stage-6 schema: {schema}")

    out_path = output_dir / args.filename
    out_path.write_text(build_html(stage6), encoding="utf-8")

    model = stage6.get("personal_model") or {}
    print("=" * 72)
    print(f"SelfTrace Stage 7 v{STAGE_VERSION}")
    print(f"Input    : {input_path}")
    print(f"Claims   : {len(model.get('claims') or [])}")
    print(f"Current  : {len(model.get('current_state') or [])}")
    print(f"Evolution: {len(model.get('evolution_patterns') or [])}")
    print(f"Output   : {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

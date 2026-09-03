#!/usr/bin/env python3
"""SelfTrace Stage 5 — Temporal Aggregation (v2.1).

Consumes Stage-4 per-post JSON files and turns isolated observations into an
Evidence-backed temporal trace. This stage intentionally DOES NOT create final
personal/profile claims; that is Stage 6.

Pipeline:
    Stage-4 JSONs
      -> evidence ledger
      -> semantic canonicalization (OpenAI or local Ollama)
      -> deterministic temporal metrics
      -> topic trajectories + signal clusters + dimension summaries
      -> temporal_aggregation.json + temporal_report.md

Design principles:
- No re-reading images.
- No personality inference.
- Preserve post_id/source_url/time/evidence for every aggregate.
- LLM is used only to canonicalize semantically similar labels/claims.
- Counts, dates, confidence weights and temporal labels are computed in Python.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_VERSION = "2.1"
SCHEMA_VERSION = "selftrace-temporal-v2.1"
LOGGER = logging.getLogger("selftrace.temporal")

load_dotenv(PROJECT_ROOT / ".env.local", override=False)


# -----------------------------------------------------------------------------
# Structured schema for semantic canonicalization
# -----------------------------------------------------------------------------

class CanonicalMapping(BaseModel):
    item_id: str
    canonical_label: str
    theme: str


class CanonicalBatch(BaseModel):
    mappings: list[CanonicalMapping] = Field(default_factory=list)


CANONICALIZE_SYSTEM = """You normalize labels for SelfTrace temporal aggregation.
You are NOT profiling the person and must not infer new traits.

For every supplied item, return exactly one mapping with the same item_id.
Your job is only to merge semantically equivalent or near-equivalent concepts.

Rules:
- canonical_label: short, concrete noun phrase; preserve the meaning of the item.
- theme: a slightly broader concrete category, not a personality judgment.
- Equivalent concepts should receive exactly the same canonical_label.
- Distinct concepts must remain distinct.
- For SIGNAL items, respect the dimension. Never merge across dimensions in a way
  that changes the meaning (e.g. interest in X is not knowledge of X).
- Do not infer sensitive attributes, diagnoses, personality, ideology, or motives.
- Do not invent evidence or new item IDs.
- Keep labels concise (usually 2-8 words).
"""


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging(log_file: Path, console_level: str = "INFO") -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    console = TqdmLoggingHandler()
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    ))
    LOGGER.addHandler(console)
    LOGGER.addHandler(fh)


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_dt(value: str) -> datetime:
    if not value:
        raise ValueError("missing datetime")
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_day(dt: datetime) -> str:
    return dt.date().isoformat()


def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_config(path: str | Path) -> dict:
    p = resolve_project_path(path)
    if not p.exists():
        return {
            "provider": "openai",
            "openai": {"model": "gpt-5.6-luna", "retries": 1},
            "local": {"model": "qwen3-vl:8b", "retries": 1},
        }
    return load_json(p)


def provider_settings(config: dict, provider_override: str | None, model_override: str | None):
    provider = (provider_override or config.get("provider") or "openai").lower()
    if provider not in {"openai", "local"}:
        raise ValueError("provider must be openai or local")
    section = dict(config.get(provider) or {})
    default_model = "gpt-5.6-luna" if provider == "openai" else "qwen3-vl:8b"
    model = model_override or section.get("model") or default_model
    retries = max(0, int(section.get("retries", 1)))
    return provider, model, retries


# -----------------------------------------------------------------------------
# Stage-4 ingestion
# -----------------------------------------------------------------------------

def discover_post_files(input_dir: Path) -> list[Path]:
    files = []
    for p in sorted(input_dir.glob("*.json")):
        if p.name.startswith("batch_summary_") or p.name.startswith("temporal_"):
            continue
        try:
            d = load_json(p)
        except Exception:
            continue
        if str(d.get("schema_version", "")).startswith("selftrace-analysis-v2") and d.get("post_id"):
            files.append(p)
    return files


def read_posts(input_dir: Path) -> list[dict]:
    posts = []
    for p in discover_post_files(input_dir):
        d = load_json(p)
        if d.get("analysis_status") not in {"OK", "OK_WITH_WARNINGS"}:
            continue
        if not d.get("created_at"):
            LOGGER.warning("Skip missing created_at | %s", p.name)
            continue
        d["_path"] = str(p)
        d["_dt"] = parse_dt(d["created_at"])
        posts.append(d)
    posts.sort(key=lambda x: x["_dt"])
    return posts


def build_ledger(posts: list[dict], include_secondary_topics: bool) -> tuple[list[dict], list[dict]]:
    """Return topic evidence records and signal evidence records."""
    topics: list[dict] = []
    signals: list[dict] = []

    for post in posts:
        pu = post.get("post_understanding") or {}
        topic_values = list(pu.get("primary_topics") or [])
        if include_secondary_topics:
            topic_values += list(pu.get("secondary_topics") or [])

        seen_topic_keys = set()
        for topic in topic_values:
            key = normalized_key(topic)
            if not key or key in seen_topic_keys:
                continue
            seen_topic_keys.add(key)
            topics.append({
                "post_id": str(post.get("post_id")),
                "created_at": post.get("created_at"),
                "source_url": post.get("source_url"),
                "topic": str(topic).strip(),
                "topic_role": "primary" if topic in (pu.get("primary_topics") or []) else "secondary",
            })

        for idx, signal in enumerate(post.get("evidence_signals") or []):
            signals.append({
                "signal_id": f"{post.get('post_id')}::s{idx}",
                "post_id": str(post.get("post_id")),
                "created_at": post.get("created_at"),
                "source_url": post.get("source_url"),
                "dimension": signal.get("dimension"),
                "claim": signal.get("claim") or "",
                "evidence": list(signal.get("evidence") or []),
                "confidence": signal.get("confidence") or "MEDIUM",
                "primary_topics": list(pu.get("primary_topics") or []),
            })

    return topics, signals


# -----------------------------------------------------------------------------
# Semantic canonicalization provider
# -----------------------------------------------------------------------------

def _extract_openai_parsed(response) -> CanonicalBatch:
    for output in response.output:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []):
            if getattr(item, "type", None) == "output_text" and getattr(item, "parsed", None) is not None:
                return item.parsed
    raise RuntimeError("No parsed canonicalization output")


def canonicalize_openai(model: str, items: list[dict], retries: int) -> CanonicalBatch:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available")
    from openai import OpenAI
    client = OpenAI()
    payload = json.dumps({"items": items}, ensure_ascii=False)
    last = None
    for attempt in range(retries + 1):
        try:
            r = client.responses.parse(
                model=model,
                instructions=CANONICALIZE_SYSTEM,
                input=[{"role": "user", "content": [{"type": "input_text", "text": payload}]}],
                text_format=CanonicalBatch,
                reasoning={"effort": "none"},
            )
            return _extract_openai_parsed(r)
        except Exception as exc:
            last = exc
            LOGGER.warning("Canonicalization OpenAI failure | attempt=%d | %s", attempt + 1, exc)
            if attempt < retries:
                time.sleep(min(2 * (attempt + 1), 5))
    raise RuntimeError(f"OpenAI canonicalization failed: {last}")


def canonicalize_local(model: str, items: list[dict], retries: int) -> CanonicalBatch:
    from ollama import chat
    payload = json.dumps({"items": items}, ensure_ascii=False)
    last = None
    for attempt in range(retries + 1):
        try:
            r = chat(
                model=model,
                format=CanonicalBatch.model_json_schema(),
                messages=[
                    {"role": "system", "content": CANONICALIZE_SYSTEM},
                    {"role": "user", "content": payload},
                ],
                options={"temperature": 0},
            )
            return CanonicalBatch.model_validate_json(r.message.content)
        except Exception as exc:
            last = exc
            LOGGER.warning("Canonicalization Ollama failure | attempt=%d | %s", attempt + 1, exc)
            if attempt < retries:
                time.sleep(min(2 * (attempt + 1), 5))
    raise RuntimeError(f"Ollama canonicalization failed: {last}")


def canonicalize_items(
    items: list[dict],
    provider: str,
    model: str,
    retries: int,
    batch_size: int,
    desc: str,
) -> dict[str, dict]:
    """Canonicalize items in batches; missing outputs fall back deterministically."""
    mapping: dict[str, dict] = {}
    if not items:
        return mapping

    chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    progress = tqdm(chunks, desc=desc, unit="batch", dynamic_ncols=True)
    for chunk in progress:
        result = (
            canonicalize_openai(model, chunk, retries)
            if provider == "openai"
            else canonicalize_local(model, chunk, retries)
        )
        expected = {x["item_id"]: x for x in chunk}
        for m in result.mappings:
            if m.item_id not in expected:
                LOGGER.warning("Ignore unknown item_id from canonicalizer: %s", m.item_id)
                continue
            label = re.sub(r"\s+", " ", m.canonical_label.strip())
            theme = re.sub(r"\s+", " ", m.theme.strip())
            if label:
                mapping[m.item_id] = {"canonical_label": label, "theme": theme or label}

        # Do not let a malformed/missing mapping break the evidence chain.
        for item_id, original in expected.items():
            if item_id not in mapping:
                fallback = original.get("text") or original.get("claim") or item_id
                mapping[item_id] = {"canonical_label": fallback.strip(), "theme": fallback.strip()}
                LOGGER.warning("Fallback canonical label | %s", item_id)
    return mapping


def prepare_unique_topic_items(topic_records: list[dict]) -> tuple[list[dict], dict[str, str]]:
    by_key: dict[str, str] = {}
    unique_items: list[dict] = []
    for rec in topic_records:
        key = normalized_key(rec["topic"])
        if key not in by_key:
            item_id = f"t{len(unique_items):04d}"
            by_key[key] = item_id
            unique_items.append({"item_id": item_id, "kind": "TOPIC", "text": rec["topic"]})
    return unique_items, by_key


def prepare_signal_items(signal_records: list[dict]) -> list[dict]:
    return [
        {
            "item_id": rec["signal_id"],
            "kind": "SIGNAL",
            "dimension": rec["dimension"],
            "claim": rec["claim"],
            "primary_topics": rec["primary_topics"],
        }
        for rec in signal_records
    ]


# -----------------------------------------------------------------------------
# Deterministic temporal metrics
# -----------------------------------------------------------------------------

CONF_WEIGHTS = {"HIGH": 1.0, "MEDIUM": 0.65}


def temporal_label(
    dates: list[datetime],
    reference: datetime,
    recent_window_days: int,
    persistent_days: int,
) -> str:
    """Simple auditable labels; no semantic interpretation is performed here."""
    dates = sorted(set(dates))
    count = len(dates)
    first_seen, last_seen = dates[0], dates[-1]
    span = (last_seen.date() - first_seen.date()).days
    recency = (reference.date() - last_seen.date()).days
    active_months = len({month_key(d) for d in dates})

    if count == 1:
        return "SINGLE_OBSERVATION"
    if recency > recent_window_days:
        return "INACTIVE_RECENTLY"
    if (reference.date() - first_seen.date()).days <= recent_window_days and span <= recent_window_days:
        return "EMERGING"
    if count >= 3 and span >= persistent_days:
        return "PERSISTENT"
    if active_months >= 2:
        return "RECURRING"
    return "REPEATED"


def base_temporal_metrics(
    records: list[dict],
    reference: datetime,
    recent_window_days: int,
    persistent_days: int,
) -> dict:
    dates = [parse_dt(r["created_at"]) for r in records]
    unique_dates = sorted(set(dates))
    post_ids = sorted({r["post_id"] for r in records})
    first, last = unique_dates[0], unique_dates[-1]
    month_counts = Counter(month_key(d) for d in dates)

    return {
        "evidence_count": len(records),
        "post_count": len(post_ids),
        "first_seen": first.isoformat().replace("+00:00", "Z"),
        "last_seen": last.isoformat().replace("+00:00", "Z"),
        "active_span_days": (last.date() - first.date()).days,
        "recency_days": (reference.date() - last.date()).days,
        "active_months": len(month_counts),
        "monthly_counts": dict(sorted(month_counts.items())),
        "temporal_label": temporal_label(unique_dates, reference, recent_window_days, persistent_days),
        "post_ids": post_ids,
    }


def aggregate_topics(
    topic_records: list[dict],
    topic_item_by_key: dict[str, str],
    canonical: dict[str, dict],
    reference: datetime,
    recent_window_days: int,
    persistent_days: int,
) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    metadata: dict[str, dict] = {}
    for rec in topic_records:
        item_id = topic_item_by_key[normalized_key(rec["topic"])]
        c = canonical[item_id]
        key = normalized_key(c["canonical_label"])
        groups[key].append(rec)
        metadata[key] = c

    out = []
    for key, records in groups.items():
        metrics = base_temporal_metrics(records, reference, recent_window_days, persistent_days)
        role_counts = Counter(r["topic_role"] for r in records)
        evidence = [
            {
                "post_id": r["post_id"],
                "created_at": r["created_at"],
                "source_url": r["source_url"],
                "observed_topic": r["topic"],
                "topic_role": r["topic_role"],
            }
            for r in sorted(records, key=lambda x: x["created_at"])
        ]
        out.append({
            "canonical_topic": metadata[key]["canonical_label"],
            "theme": metadata[key]["theme"],
            **metrics,
            "role_counts": dict(role_counts),
            "evidence": evidence,
        })

    out.sort(key=lambda x: (-x["post_count"], -x["evidence_count"], x["canonical_topic"].lower()))
    return out


def aggregate_signals(
    signal_records: list[dict],
    canonical: dict[str, dict],
    reference: datetime,
    recent_window_days: int,
    persistent_days: int,
) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    metadata: dict[tuple[str, str], dict] = {}

    for rec in signal_records:
        c = canonical[rec["signal_id"]]
        key = (rec["dimension"], normalized_key(c["canonical_label"]))
        groups[key].append(rec)
        metadata[key] = c

    out = []
    for (dimension, _), records in groups.items():
        metrics = base_temporal_metrics(records, reference, recent_window_days, persistent_days)
        conf_counts = Counter(r["confidence"] for r in records)
        weights = [CONF_WEIGHTS.get(r["confidence"], 0.5) for r in records]
        confidence_score = sum(weights) / len(weights) if weights else 0.0
        group_key = (dimension, _)
        c = metadata[group_key]

        evidence = [
            {
                "signal_id": r["signal_id"],
                "post_id": r["post_id"],
                "created_at": r["created_at"],
                "source_url": r["source_url"],
                "claim": r["claim"],
                "evidence": r["evidence"],
                "confidence": r["confidence"],
                "primary_topics": r["primary_topics"],
            }
            for r in sorted(records, key=lambda x: x["created_at"])
        ]
        out.append({
            "dimension": dimension,
            "canonical_signal": c["canonical_label"],
            "theme": c["theme"],
            **metrics,
            "confidence_counts": dict(conf_counts),
            "mean_evidence_strength": round(confidence_score, 3),
            "evidence": evidence,
        })

    out.sort(key=lambda x: (x["dimension"], -x["post_count"], -x["mean_evidence_strength"], x["canonical_signal"].lower()))
    return out


def dimension_summary(signal_records: list[dict], reference: datetime) -> list[dict]:
    groups = defaultdict(list)
    for r in signal_records:
        groups[r["dimension"]].append(r)
    out = []
    for dimension, records in sorted(groups.items()):
        dates = [parse_dt(r["created_at"]) for r in records]
        counts = Counter(r["confidence"] for r in records)
        monthly = Counter(month_key(d) for d in dates)
        out.append({
            "dimension": dimension,
            "signal_count": len(records),
            "post_count": len({r["post_id"] for r in records}),
            "high_confidence": counts.get("HIGH", 0),
            "medium_confidence": counts.get("MEDIUM", 0),
            "first_seen": min(dates).isoformat().replace("+00:00", "Z"),
            "last_seen": max(dates).isoformat().replace("+00:00", "Z"),
            "recency_days": (reference.date() - max(dates).date()).days,
            "monthly_counts": dict(sorted(monthly.items())),
        })
    return out


def build_timeline(posts: list[dict], topic_clusters: list[dict], signal_clusters: list[dict]) -> list[dict]:
    topic_by_post = defaultdict(list)
    for cluster in topic_clusters:
        for ev in cluster["evidence"]:
            topic_by_post[ev["post_id"]].append(cluster["canonical_topic"])

    signal_by_post = defaultdict(list)
    for cluster in signal_clusters:
        for ev in cluster["evidence"]:
            signal_by_post[ev["post_id"]].append({
                "dimension": cluster["dimension"],
                "canonical_signal": cluster["canonical_signal"],
                "confidence": ev["confidence"],
            })

    return [
        {
            "post_id": str(p["post_id"]),
            "created_at": p["created_at"],
            "source_url": p.get("source_url"),
            "raw_text": p.get("raw_text") or "",
            "canonical_topics": sorted(set(topic_by_post[str(p["post_id"])])),
            "signals": signal_by_post[str(p["post_id"])],
        }
        for p in posts
    ]


# -----------------------------------------------------------------------------
# Human-readable report
# -----------------------------------------------------------------------------

def md_escape(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def make_report(result: dict) -> str:
    ds = result["dataset"]
    lines = [
        "# SelfTrace Stage 5 — Temporal Aggregation",
        "",
        "> This report aggregates Stage-4 evidence. It does **not** create final personality/profile claims.",
        "",
        "## Dataset",
        "",
        f"- Posts: **{ds['post_count']}**",
        f"- Signals: **{ds['signal_count']}**",
        f"- Time range: **{ds['first_seen']} → {ds['last_seen']}**",
        f"- Provider/model for semantic normalization: **{result['canonicalization']['provider']} / {result['canonicalization']['model']}**",
        "",
        "## Dimension Summary",
        "",
        "| Dimension | Signals | Posts | HIGH | MEDIUM | First | Last |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for d in result["dimension_summary"]:
        lines.append(
            f"| {d['dimension']} | {d['signal_count']} | {d['post_count']} | "
            f"{d['high_confidence']} | {d['medium_confidence']} | {d['first_seen'][:10]} | {d['last_seen'][:10]} |"
        )

    lines += ["", "## Topic Trajectories", "",
              "| Topic | Theme | Posts | Span days | Recency | Temporal label |",
              "|---|---|---:|---:|---:|---|"]
    for t in result["topic_trajectories"][:30]:
        lines.append(
            f"| {md_escape(t['canonical_topic'])} | {md_escape(t['theme'])} | {t['post_count']} | "
            f"{t['active_span_days']} | {t['recency_days']} | {t['temporal_label']} |"
        )

    lines += ["", "## Signal Clusters", ""]
    by_dim = defaultdict(list)
    for s in result["signal_clusters"]:
        by_dim[s["dimension"]].append(s)
    for dim in sorted(by_dim):
        lines += [f"### {dim}", "", "| Signal | Theme | Posts | Strength | Temporal label |",
                  "|---|---|---:|---:|---|"]
        for s in by_dim[dim][:20]:
            lines.append(
                f"| {md_escape(s['canonical_signal'])} | {md_escape(s['theme'])} | {s['post_count']} | "
                f"{s['mean_evidence_strength']:.3f} | {s['temporal_label']} |"
            )
        lines.append("")

    lines += [
        "## How to read temporal labels",
        "",
        "- `SINGLE_OBSERVATION`: observed once.",
        "- `EMERGING`: repeated and first observed within the recent window.",
        "- `PERSISTENT`: repeated across a sufficiently long span.",
        "- `RECURRING`: repeated across multiple months but below the persistence threshold.",
        "- `REPEATED`: repeated within a shorter interval.",
        "- `INACTIVE_RECENTLY`: repeated historically but not observed in the recent window.",
        "",
        "These are deterministic Stage-5 temporal descriptors, not personality judgments.",
        "",
        "## Next stage",
        "",
        "Stage 6 can consume these evidence-backed clusters to generate higher-level Personal Model claims.",
    ]
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SelfTrace Stage 5 temporal aggregation v2.1")
    p.add_argument("--input-dir", default="data/analysis/openai", type=Path)
    p.add_argument("--output-dir", default="data/temporal", type=Path)
    p.add_argument("--config", default="config.json", type=Path)
    p.add_argument("--provider", choices=["openai", "local"], default=None,
                   help="Provider used only for semantic canonicalization")
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=60)
    p.add_argument("--include-secondary-topics", action="store_true")
    p.add_argument("--recent-window-days", type=int, default=30)
    p.add_argument("--persistent-days", type=int, default=60)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = PROJECT_ROOT / "logs" / f"temporal_aggregate_{timestamp}.log"
    setup_logging(log_file, args.log_level)

    config = load_config(args.config)
    provider, model, retries = provider_settings(config, args.provider, args.model)

    LOGGER.info("=" * 86)
    LOGGER.info("SelfTrace Stage 5 v%s", STAGE_VERSION)
    LOGGER.info("Input dir    : %s", input_dir)
    LOGGER.info("Output dir   : %s", output_dir)
    LOGGER.info("Provider     : %s", provider)
    LOGGER.info("Model        : %s", model)
    LOGGER.info("Recent window: %d days", args.recent_window_days)
    LOGGER.info("Persistence  : %d days", args.persistent_days)
    LOGGER.info("=" * 86)

    try:
        posts = read_posts(input_dir)
        if not posts:
            raise RuntimeError(f"No Stage-4 post analysis JSON files found in {input_dir}")

        topic_records, signal_records = build_ledger(posts, args.include_secondary_topics)
        reference = max(p["_dt"] for p in posts)
        first = min(p["_dt"] for p in posts)

        LOGGER.info("Posts loaded : %d", len(posts))
        LOGGER.info("Topics       : %d evidence records", len(topic_records))
        LOGGER.info("Signals      : %d evidence records", len(signal_records))
        LOGGER.info("Time range   : %s -> %s", iso_day(first), iso_day(reference))

        # Canonicalize topics: dedupe exact labels first to minimize token use.
        unique_topics, topic_item_by_key = prepare_unique_topic_items(topic_records)
        LOGGER.info("Unique topic strings: %d", len(unique_topics))
        topic_map = canonicalize_items(
            unique_topics, provider, model, retries, max(1, args.batch_size), "Canonicalize topics"
        )

        # Canonicalize each signal claim; the dimension is always included.
        signal_items = prepare_signal_items(signal_records)
        signal_map = canonicalize_items(
            signal_items, provider, model, retries, max(1, args.batch_size), "Canonicalize signals"
        )

        topic_clusters = aggregate_topics(
            topic_records, topic_item_by_key, topic_map, reference,
            args.recent_window_days, args.persistent_days,
        )
        signal_clusters = aggregate_signals(
            signal_records, signal_map, reference,
            args.recent_window_days, args.persistent_days,
        )
        dims = dimension_summary(signal_records, reference)
        timeline = build_timeline(posts, topic_clusters, signal_clusters)

        result = {
            "schema_version": SCHEMA_VERSION,
            "stage_version": STAGE_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "stage": "TEMPORAL_AGGREGATION",
            "stage_boundary": {
                "does": [
                    "aggregate Stage-4 evidence over time",
                    "canonicalize semantically similar topics and signal claims",
                    "compute deterministic frequency, span, recency, monthly counts and temporal labels",
                    "preserve evidence provenance",
                ],
                "does_not": [
                    "re-analyze images",
                    "infer personality",
                    "generate final Personal Model claims",
                ],
            },
            "dataset": {
                "post_count": len(posts),
                "signal_count": len(signal_records),
                "topic_evidence_count": len(topic_records),
                "first_seen": first.isoformat().replace("+00:00", "Z"),
                "last_seen": reference.isoformat().replace("+00:00", "Z"),
                "span_days": (reference.date() - first.date()).days,
                "input_dir": str(input_dir),
            },
            "canonicalization": {
                "provider": provider,
                "model": model,
                "topic_input_count": len(unique_topics),
                "signal_input_count": len(signal_items),
            },
            "temporal_rules": {
                "recent_window_days": args.recent_window_days,
                "persistent_days": args.persistent_days,
                "confidence_weights": CONF_WEIGHTS,
            },
            "dimension_summary": dims,
            "topic_trajectories": topic_clusters,
            "signal_clusters": signal_clusters,
            "timeline": timeline,
            "evidence_ledger": {
                "topics": topic_records,
                "signals": signal_records,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "log_file": str(log_file),
        }

        json_path = output_dir / "temporal_aggregation.json"
        md_path = output_dir / "temporal_report.md"
        save_json(result, json_path)
        md_path.write_text(make_report(result), encoding="utf-8")

        LOGGER.info("=" * 86)
        LOGGER.info("Stage 5 completed")
        LOGGER.info("Topic clusters : %d", len(topic_clusters))
        LOGGER.info("Signal clusters: %d", len(signal_clusters))
        LOGGER.info("JSON           : %s", json_path)
        LOGGER.info("Report         : %s", md_path)
        LOGGER.info("Elapsed        : %.2fs", time.perf_counter() - started)
        LOGGER.info("=" * 86)
    except Exception as exc:
        LOGGER.error("Stage 5 failed | %s", exc)
        LOGGER.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

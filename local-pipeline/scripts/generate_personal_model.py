#!/usr/bin/env python3
"""SelfTrace Stage 6 — Personal Model Generation (v2.2).

Consumes Stage-5 temporal_aggregation.json and synthesizes an evidence-backed
Personal Model. This stage does NOT re-read posts or images and does NOT perform
new low-level extraction.

Pipeline:
    Stage-5 temporal_aggregation.json
      -> compact temporal evidence package
      -> one LLM synthesis call
      -> validate source references
      -> expand evidence provenance
      -> personal_model.json + personal_model.md

The model is intentionally descriptive rather than diagnostic. It models
observed interests, knowledge, activities, thinking operations, expression
patterns, goals, current state and temporal evolution. It must not infer
sensitive attributes or personality diagnoses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_VERSION = "2.2"
SCHEMA_VERSION = "selftrace-personal-model-v2.2"
LOGGER = logging.getLogger("selftrace.personal_model")

load_dotenv(PROJECT_ROOT / ".env.local", override=False)


# -----------------------------------------------------------------------------
# Output schema requested from the synthesis model
# -----------------------------------------------------------------------------

Dimension = Literal[
    "INTEREST",
    "KNOWLEDGE",
    "ACTIVITY",
    "THINKING_PATTERN",
    "EXPRESSION_STYLE",
    "GOAL_ORIENTATION",
]

Domain = Literal[
    "COGNITIVE",
    "MOTIVATIONAL",
    "BEHAVIORAL",
    "COMMUNICATIVE",
]

Scope = Literal[
    "LONG_TERM",
    "RECURRING",
    "CURRENT",
    "EMERGING",
    "HISTORICAL",
]

Confidence = Literal["HIGH", "MEDIUM"]


class PersonalClaim(BaseModel):
    domain: Domain
    dimension: Dimension
    statement: str
    scope: Scope
    confidence: Confidence
    rationale: str
    evidence_ref_ids: list[str] = Field(default_factory=list)


class CurrentStateItem(BaseModel):
    label: str
    kind: Literal["FOCUS", "INTEREST", "ACTIVITY", "GOAL", "KNOWLEDGE_AREA"]
    confidence: Confidence
    rationale: str
    evidence_ref_ids: list[str] = Field(default_factory=list)


class EvolutionPattern(BaseModel):
    statement: str
    start_period: str | None = None
    end_period: str | None = None
    confidence: Confidence
    rationale: str
    evidence_ref_ids: list[str] = Field(default_factory=list)


class PersonalModelSynthesis(BaseModel):
    overall_summary: str
    claims: list[PersonalClaim] = Field(default_factory=list)
    current_state: list[CurrentStateItem] = Field(default_factory=list)
    evolution_patterns: list[EvolutionPattern] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


SYNTHESIS_SYSTEM = """You are the Stage-6 synthesis engine for SelfTrace.
You receive ONLY Stage-5 temporal aggregates derived from a person's own digital traces.
Your task is to build a conservative, evidence-backed Personal Model.

The Personal Model is NOT a personality diagnosis. It is a structured model of
observed interests, knowledge, activities, goals, thinking operations, expression
patterns, current focus and changes over time.

STRICT EVIDENCE RULES
- Use only the supplied evidence records. Never use outside knowledge about the person.
- Every claim/current-state/evolution item MUST cite one or more valid evidence_ref_ids.
- Do not invent IDs.
- A LONG_TERM or RECURRING claim normally needs evidence from at least 2 distinct posts.
- A single recent explicit observation may support CURRENT or EMERGING, but not a stable trait.
- Do not turn topic exposure into knowledge. KNOWLEDGE claims require KNOWLEDGE evidence.
- Do not turn activity into interest unless INTEREST evidence exists.
- Do not turn explanation style into a cognitive trait unless THINKING_PATTERN evidence exists.
- Prefer omission over weak inference.

SEMANTIC RULES
- INTEREST: what repeatedly attracts attention or is explicitly liked/valued.
- KNOWLEDGE: demonstrated understanding or competence.
- ACTIVITY: what the person actually did or was doing.
- THINKING_PATTERN: repeated reasoning operations, not personality adjectives.
- EXPRESSION_STYLE: recurring ways of communicating or structuring information.
- GOAL_ORIENTATION: explicit aims, intentions or plans.

SCOPE RULES
- LONG_TERM: supported across a meaningful time span, not a single burst.
- RECURRING: observed more than once across separated times/contexts.
- CURRENT: supported in the recent period.
- EMERGING: recent signal with limited history.
- HISTORICAL: supported in older evidence but not recent.

DOMAIN MAPPING
- COGNITIVE: KNOWLEDGE, THINKING_PATTERN
- MOTIVATIONAL: INTEREST, GOAL_ORIENTATION
- BEHAVIORAL: ACTIVITY
- COMMUNICATIVE: EXPRESSION_STYLE

SAFETY / BOUNDARY
- Never infer politics, religion, sexuality, health/diagnosis, ethnicity, relationships,
  protected traits, or other sensitive attributes.
- Never infer IQ, mental disorders, or fixed personality types.
- Never claim the model represents the whole person; it represents observed digital traces.

OUTPUT QUALITY
- Keep the model compact and useful: normally 8-20 claims total.
- Prefer cross-post patterns over one-off details.
- Current state may include 3-8 items.
- Evolution patterns should describe actual observed shifts or continuities, not storytelling.
"""


# -----------------------------------------------------------------------------
# Helpers
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


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------------------------
# Compact Stage-5 evidence package
# -----------------------------------------------------------------------------

def compact_temporal(stage5: dict) -> tuple[dict, dict[str, dict]]:
    """Strip heavy raw evidence while assigning stable IDs for synthesis.

    The model sees aggregate metrics and claim labels. The full evidence stays in
    Python and is re-attached after synthesis using evidence_ref_ids.
    """
    ref_map: dict[str, dict] = {}
    topics_out: list[dict] = []
    signals_out: list[dict] = []

    for i, topic in enumerate(stage5.get("topic_trajectories") or [], 1):
        ref_id = f"T{i:04d}"
        ref_map[ref_id] = {"source_type": "TOPIC_TRAJECTORY", "record": topic}
        topics_out.append({
            "ref_id": ref_id,
            "canonical_topic": topic.get("canonical_topic"),
            "theme": topic.get("theme"),
            "post_count": topic.get("post_count", 0),
            "evidence_count": topic.get("evidence_count", 0),
            "first_seen": topic.get("first_seen"),
            "last_seen": topic.get("last_seen"),
            "active_span_days": topic.get("active_span_days", 0),
            "recency_days": topic.get("recency_days"),
            "active_months": topic.get("active_months", 0),
            "monthly_counts": topic.get("monthly_counts") or {},
            "temporal_label": topic.get("temporal_label"),
            "post_ids": topic.get("post_ids") or [],
        })

    for i, signal in enumerate(stage5.get("signal_clusters") or [], 1):
        ref_id = f"S{i:04d}"
        ref_map[ref_id] = {"source_type": "SIGNAL_CLUSTER", "record": signal}
        signals_out.append({
            "ref_id": ref_id,
            "dimension": signal.get("dimension"),
            "canonical_signal": signal.get("canonical_signal"),
            "theme": signal.get("theme"),
            "post_count": signal.get("post_count", 0),
            "evidence_count": signal.get("evidence_count", 0),
            "first_seen": signal.get("first_seen"),
            "last_seen": signal.get("last_seen"),
            "active_span_days": signal.get("active_span_days", 0),
            "recency_days": signal.get("recency_days"),
            "active_months": signal.get("active_months", 0),
            "monthly_counts": signal.get("monthly_counts") or {},
            "temporal_label": signal.get("temporal_label"),
            "confidence_counts": signal.get("confidence_counts") or {},
            "mean_evidence_strength": signal.get("mean_evidence_strength"),
            "post_ids": signal.get("post_ids") or [],
        })

    compact = {
        "dataset": stage5.get("dataset") or {},
        "dimension_summary": stage5.get("dimension_summary") or [],
        "temporal_rules": stage5.get("temporal_rules") or {},
        "topic_trajectories": topics_out,
        "signal_clusters": signals_out,
    }
    return compact, ref_map


def build_user_prompt(compact: dict) -> str:
    return (
        "Build the Stage-6 Personal Model from this Stage-5 temporal evidence package. "
        "Use evidence_ref_ids exactly as provided.\n\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    )


# -----------------------------------------------------------------------------
# Provider calls
# -----------------------------------------------------------------------------

def extract_openai_parsed(response) -> PersonalModelSynthesis:
    for output in response.output:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []):
            if getattr(item, "type", None) == "output_text" and getattr(item, "parsed", None) is not None:
                return item.parsed
    raise RuntimeError("OpenAI response completed but no parsed Personal Model was found")


def call_openai(model: str, prompt: str, retries: int) -> tuple[PersonalModelSynthesis, dict]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable. Keep it in .env.local or your shell environment.")

    from openai import OpenAI
    client = OpenAI()
    last_error: Exception | None = None

    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        LOGGER.info("OpenAI Stage-6 start | attempt=%d/%d | model=%s", attempt, retries + 1, model)
        try:
            response = client.responses.parse(
                model=model,
                instructions=SYNTHESIS_SYSTEM,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                text_format=PersonalModelSynthesis,
                reasoning={"effort": "none"},
            )
            parsed = extract_openai_parsed(response)
            usage_obj = getattr(response, "usage", None)
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", None),
                "output_tokens": getattr(usage_obj, "output_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
            LOGGER.info("OpenAI Stage-6 done | %.2fs | tokens=%s", time.perf_counter() - started, usage)
            return parsed, usage
        except Exception as exc:
            last_error = exc
            LOGGER.warning("OpenAI Stage-6 failed | attempt=%d | %s", attempt, exc)
            LOGGER.debug(traceback.format_exc())
            if attempt <= retries:
                time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(f"OpenAI Stage-6 failed after {retries + 1} attempts: {last_error}")


def call_local(model: str, prompt: str, retries: int) -> tuple[PersonalModelSynthesis, dict]:
    from ollama import chat
    last_error: Exception | None = None

    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        LOGGER.info("Ollama Stage-6 start | attempt=%d/%d | model=%s", attempt, retries + 1, model)
        try:
            response = chat(
                model=model,
                format=PersonalModelSynthesis.model_json_schema(),
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0},
            )
            parsed = PersonalModelSynthesis.model_validate_json(response.message.content)
            LOGGER.info("Ollama Stage-6 done | %.2fs", time.perf_counter() - started)
            return parsed, {}
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Ollama Stage-6 failed | attempt=%d | %s", attempt, exc)
            LOGGER.debug(traceback.format_exc())
            if attempt <= retries:
                time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(f"Ollama Stage-6 failed after {retries + 1} attempts: {last_error}")


# -----------------------------------------------------------------------------
# Provenance validation / expansion
# -----------------------------------------------------------------------------

def validate_refs(ref_ids: list[str], ref_map: dict[str, dict]) -> tuple[list[str], list[str]]:
    valid, invalid = [], []
    seen = set()
    for ref_id in ref_ids:
        if ref_id in seen:
            continue
        seen.add(ref_id)
        if ref_id in ref_map:
            valid.append(ref_id)
        else:
            invalid.append(ref_id)
    return valid, invalid


def expand_refs(ref_ids: list[str], ref_map: dict[str, dict]) -> list[dict]:
    out = []
    for ref_id in ref_ids:
        entry = ref_map.get(ref_id)
        if not entry:
            continue
        rec = entry["record"]
        base = {
            "ref_id": ref_id,
            "source_type": entry["source_type"],
            "post_ids": rec.get("post_ids") or [],
            "first_seen": rec.get("first_seen"),
            "last_seen": rec.get("last_seen"),
            "temporal_label": rec.get("temporal_label"),
        }
        if entry["source_type"] == "SIGNAL_CLUSTER":
            base.update({
                "dimension": rec.get("dimension"),
                "canonical_signal": rec.get("canonical_signal"),
                "theme": rec.get("theme"),
                "evidence": rec.get("evidence") or [],
            })
        else:
            base.update({
                "canonical_topic": rec.get("canonical_topic"),
                "theme": rec.get("theme"),
                "evidence": rec.get("evidence") or [],
            })
        out.append(base)
    return out


def normalize_synthesis(synthesis: PersonalModelSynthesis, ref_map: dict[str, dict]) -> tuple[dict, dict]:
    report = {
        "invalid_reference_count": 0,
        "invalid_references": [],
        "dropped_items_without_valid_evidence": 0,
    }

    claims = []
    for idx, claim in enumerate(synthesis.claims, 1):
        valid, invalid = validate_refs(claim.evidence_ref_ids, ref_map)
        report["invalid_reference_count"] += len(invalid)
        report["invalid_references"].extend(invalid)
        if not valid:
            report["dropped_items_without_valid_evidence"] += 1
            continue
        d = claim.model_dump()
        d["claim_id"] = f"C{len(claims)+1:03d}"
        d["evidence_ref_ids"] = valid
        d["evidence_trace"] = expand_refs(valid, ref_map)
        claims.append(d)

    current_state = []
    for item in synthesis.current_state:
        valid, invalid = validate_refs(item.evidence_ref_ids, ref_map)
        report["invalid_reference_count"] += len(invalid)
        report["invalid_references"].extend(invalid)
        if not valid:
            report["dropped_items_without_valid_evidence"] += 1
            continue
        d = item.model_dump()
        d["evidence_ref_ids"] = valid
        d["evidence_trace"] = expand_refs(valid, ref_map)
        current_state.append(d)

    evolution = []
    for item in synthesis.evolution_patterns:
        valid, invalid = validate_refs(item.evidence_ref_ids, ref_map)
        report["invalid_reference_count"] += len(invalid)
        report["invalid_references"].extend(invalid)
        if not valid:
            report["dropped_items_without_valid_evidence"] += 1
            continue
        d = item.model_dump()
        d["evidence_ref_ids"] = valid
        d["evidence_trace"] = expand_refs(valid, ref_map)
        evolution.append(d)

    report["invalid_references"] = sorted(set(report["invalid_references"]))

    domains = defaultdict(list)
    dimensions = defaultdict(list)
    for claim in claims:
        domains[claim["domain"]].append(claim["claim_id"])
        dimensions[claim["dimension"]].append(claim["claim_id"])

    normalized = {
        "overall_summary": synthesis.overall_summary,
        "claims": claims,
        "domain_index": dict(domains),
        "dimension_index": dict(dimensions),
        "current_state": current_state,
        "evolution_patterns": evolution,
        "limitations": synthesis.limitations,
    }
    return normalized, report


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------

def markdown_report(output: dict) -> str:
    model = output["personal_model"]
    lines = [
        "# SelfTrace Personal Model",
        "",
        f"Generated: {output['generated_at']}",
        f"Provider: {output['provider']} / {output['model']}",
        "",
        "## Overall Summary",
        "",
        model.get("overall_summary") or "",
        "",
    ]

    domain_order = ["COGNITIVE", "MOTIVATIONAL", "BEHAVIORAL", "COMMUNICATIVE"]
    claim_by_id = {c["claim_id"]: c for c in model.get("claims", [])}

    for domain in domain_order:
        ids = model.get("domain_index", {}).get(domain, [])
        if not ids:
            continue
        lines += [f"## {domain.title()} Model", ""]
        for cid in ids:
            c = claim_by_id[cid]
            posts = sorted({pid for e in c.get("evidence_trace", []) for pid in e.get("post_ids", [])})
            lines += [
                f"### {cid} — {c['statement']}",
                "",
                f"- Dimension: `{c['dimension']}`",
                f"- Scope: `{c['scope']}`",
                f"- Confidence: `{c['confidence']}`",
                f"- Rationale: {c['rationale']}",
                f"- Evidence refs: {', '.join(c['evidence_ref_ids'])}",
                f"- Supporting posts: {', '.join(posts)}",
                "",
            ]

    if model.get("current_state"):
        lines += ["## Current / Emerging State", ""]
        for item in model["current_state"]:
            lines.append(f"- **{item['label']}** — {item['kind']} / {item['confidence']}: {item['rationale']}")
        lines.append("")

    if model.get("evolution_patterns"):
        lines += ["## Evolution Patterns", ""]
        for item in model["evolution_patterns"]:
            period = " → ".join(x for x in [item.get("start_period"), item.get("end_period")] if x)
            suffix = f" ({period})" if period else ""
            lines.append(f"- **{item['statement']}**{suffix} — {item['confidence']}: {item['rationale']}")
        lines.append("")

    lines += ["## Boundaries / Limitations", ""]
    limits = model.get("limitations") or [
        "This model represents observed digital traces, not the whole person.",
        "Absence of evidence is not evidence of absence.",
    ]
    for limit in limits:
        lines.append(f"- {limit}")

    return "\n".join(lines).rstrip() + "\n"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SelfTrace Stage 6 — Personal Model Generation v2.2")
    p.add_argument("--input", default="data/temporal/temporal_aggregation.json", type=Path)
    p.add_argument("--config", default="config.json", type=Path)
    p.add_argument("--provider", choices=["openai", "local"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--output-dir", default="data/personal", type=Path)
    p.add_argument("--dry-run", action="store_true", help="Build compact Stage-6 input but do not call a model")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (PROJECT_ROOT / "logs" / f"personal_model_{timestamp}.log").resolve()
    setup_logging(log_file, args.log_level)

    config = load_config(args.config)
    provider, model, retries = provider_settings(config, args.provider, args.model)

    LOGGER.info("=" * 84)
    LOGGER.info("SelfTrace Stage 6 v%s", STAGE_VERSION)
    LOGGER.info("Input        : %s", input_path)
    LOGGER.info("Output dir   : %s", output_dir)
    LOGGER.info("Provider     : %s", provider)
    LOGGER.info("Model        : %s", model)
    LOGGER.info("=" * 84)

    stage5 = load_json(input_path)
    if not str(stage5.get("schema_version", "")).startswith("selftrace-temporal-v2"):
        raise ValueError(f"Unsupported Stage-5 schema: {stage5.get('schema_version')}")

    compact, ref_map = compact_temporal(stage5)
    preview_path = output_dir / "stage6_input_preview.json"
    save_json(compact, preview_path)

    LOGGER.info("Stage-5 posts         : %s", (stage5.get("dataset") or {}).get("post_count"))
    LOGGER.info("Topic trajectories    : %d", len(compact["topic_trajectories"]))
    LOGGER.info("Signal clusters       : %d", len(compact["signal_clusters"]))
    LOGGER.info("Evidence refs         : %d", len(ref_map))

    if args.dry_run:
        LOGGER.info("Dry run complete | preview=%s", preview_path)
        return

    prompt = build_user_prompt(compact)
    if provider == "openai":
        synthesis, usage = call_openai(model, prompt, retries)
    else:
        synthesis, usage = call_local(model, prompt, retries)

    personal_model, validator_report = normalize_synthesis(synthesis, ref_map)

    # Always include system-level boundaries even if the synthesis omitted them.
    required_limits = [
        "This Personal Model represents patterns observed in the available digital traces, not the whole person.",
        "Absence of evidence in the trace should not be interpreted as absence of an interest, capability, activity, or goal.",
        "Claims are descriptive and evidence-backed; they are not personality or psychological diagnoses.",
    ]
    existing = set(personal_model.get("limitations") or [])
    personal_model["limitations"] = list(personal_model.get("limitations") or [])
    for item in required_limits:
        if item not in existing:
            personal_model["limitations"].append(item)

    output = {
        "schema_version": SCHEMA_VERSION,
        "stage_version": STAGE_VERSION,
        "stage": "PERSONAL_MODEL_GENERATION",
        "generated_at": iso_now(),
        "provider": provider,
        "model": model,
        "source_stage5": {
            "schema_version": stage5.get("schema_version"),
            "generated_at": stage5.get("generated_at"),
            "dataset": stage5.get("dataset") or {},
        },
        "stage_boundary": {
            "does": [
                "synthesize high-level personal-model claims from Stage-5 temporal evidence",
                "distinguish long-term, recurring, current, emerging and historical observations",
                "preserve evidence references and supporting post provenance",
            ],
            "does_not": [
                "re-read raw images",
                "perform new Stage-4 signal extraction",
                "infer sensitive attributes or psychological diagnoses",
            ],
        },
        "personal_model": personal_model,
        "validator_report": validator_report,
        "usage": usage,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "log_file": str(log_file),
    }

    json_path = output_dir / "personal_model.json"
    md_path = output_dir / "personal_model.md"
    save_json(output, json_path)
    md_path.write_text(markdown_report(output), encoding="utf-8")

    LOGGER.info("Stage 6 completed | claims=%d | current=%d | evolution=%d",
                len(personal_model.get("claims", [])),
                len(personal_model.get("current_state", [])),
                len(personal_model.get("evolution_patterns", [])))
    LOGGER.info("Invalid refs           : %d", validator_report["invalid_reference_count"])
    LOGGER.info("Personal model JSON    : %s", json_path)
    LOGGER.info("Readable report        : %s", md_path)
    LOGGER.info("Elapsed                : %.2fs", time.perf_counter() - started)


if __name__ == "__main__":
    main()

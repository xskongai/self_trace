#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field

# scripts/analyze_post.py -> local-pipeline/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against local-pipeline/, never against current cwd."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


MEDIA_TYPES = Literal[
    "PHOTO", "SCREENSHOT", "PAPER_SCREENSHOT", "CHART", "TABLE", "DIAGRAM",
    "CODE", "FORMULA", "NOTE", "MEME", "OTHER"
]
POST_TYPES = Literal[
    "LEARNING_NOTE", "RESEARCH_NOTE", "PERSONAL_REFLECTION", "PROJECT_UPDATE",
    "QUESTION", "OPINION", "OBSERVATION", "PHOTO_SHARE", "RESOURCE_SHARE", "OTHER"
]
INTENTS = Literal[
    "RECORD_IDEA", "SHARE_RESULT", "ASK_QUESTION", "REFLECT", "LEARN", "EXPLAIN",
    "COMPARE", "EXPRESS_FEELING", "DOCUMENT_PROGRESS", "OTHER"
]
DIMENSIONS = Literal[
    "INTEREST", "KNOWLEDGE", "ACTIVITY", "THINKING_PATTERN", "EXPRESSION_STYLE", "GOAL_ORIENTATION"
]
CONFIDENCE = Literal["HIGH", "MEDIUM"]


class ChartObservation(BaseModel):
    title: str | None = None
    chart_type: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    entities: list[str] = Field(default_factory=list)
    main_pattern: str | None = None
    notable_values: list[str] = Field(default_factory=list)


class MediaObservation(BaseModel):
    media_type: MEDIA_TYPES
    description: str
    visible_text: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    visual_elements: list[str] = Field(default_factory=list)
    chart: ChartObservation | None = None
    uncertainty: list[str] = Field(default_factory=list)


class PostUnderstanding(BaseModel):
    content_summary: str
    primary_topics: list[str] = Field(default_factory=list)
    secondary_topics: list[str] = Field(default_factory=list)
    post_types: list[POST_TYPES] = Field(default_factory=list)
    intents: list[INTENTS] = Field(default_factory=list)
    source_relationship: Literal[
        "AUTHOR_ORIGINAL", "COMMENTARY_ON_EXTERNAL_SOURCE",
        "SHARED_SOURCE_WITHOUT_CLEAR_COMMENTARY", "UNCLEAR"
    ]
    explicit_statements: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class EvidenceSignal(BaseModel):
    dimension: DIMENSIONS
    claim: str
    evidence: list[str]
    confidence: CONFIDENCE
    temporal_scope: Literal["THIS_POST_ONLY"] = "THIS_POST_ONLY"


class SignalBundle(BaseModel):
    signals: list[EvidenceSignal] = Field(default_factory=list)
    rejected_inferences: list[str] = Field(default_factory=list)


SYSTEM_MEDIA = """You are the observation layer of SelfTrace, a personal digital-trace analysis system.
Describe only what is supported by the image. Do not infer the author's personality, mental state, political beliefs, religion, health, sexuality, ethnicity, or relationships. Do not guess unreadable text or values. If uncertain, record uncertainty. For charts/tables, extract structure and trends only when legible."""

SYSTEM_POST = """You are the post-understanding layer of SelfTrace.
Synthesize the post text and already-produced media observations. Separate the author's explicit statement from external material they may be quoting, reading, or sharing. Do not infer stable personality traits. Do not infer sensitive attributes. If evidence is ambiguous, say so."""

SYSTEM_SIGNAL = """You are the evidence-signal layer of SelfTrace.
Generate only narrow, evidence-traceable signals supported by THIS post. HIGH means directly explicit. MEDIUM means a reasonable behavioral/content inference from this post. Do not create LOW-confidence speculation; reject it instead. Never infer sensitive attributes, diagnoses, romantic feelings, or stable personality from one post. A thinking-pattern claim must be phrased as behavior shown in this post, not as a permanent trait."""


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_post(data: dict, post_id: str | None) -> dict:
    posts = data.get("posts", [])
    if post_id:
        for p in posts:
            if str(p.get("post_id")) == str(post_id):
                return p
        raise SystemExit(f"Post not found: {post_id}")
    for p in posts:
        if p.get("media"):
            return p
    raise SystemExit("No post with media found")


def analyze_media(model: str, path: Path, post_text: str) -> MediaObservation:
    prompt = f"""Analyze this single image as evidence attached to a social-media post.
Post text for context (do not let it override what you can actually see):
{post_text or '[empty]'}

Return an objective media observation. Preserve visible technical terminology. If this is a paper screenshot, distinguish paper/source content from the post author's own words. If text is too small or unclear, do not reconstruct it."""
    response = chat(
        model=model,
        format=MediaObservation.model_json_schema(),
        messages=[
            {"role": "system", "content": SYSTEM_MEDIA},
            {"role": "user", "content": prompt, "images": [str(path)]},
        ],
        options={"temperature": 0},
    )
    return MediaObservation.model_validate_json(response.message.content)


def analyze_post_understanding(model: str, post: dict, media_obs: list[dict]) -> PostUnderstanding:
    payload = {
        "post_id": str(post.get("post_id")),
        "created_at": post.get("created_at"),
        "text": post.get("text") or "",
        "media_observations": media_obs,
    }
    response = chat(
        model=model,
        format=PostUnderstanding.model_json_schema(),
        messages=[
            {"role": "system", "content": SYSTEM_POST},
            {"role": "user", "content": "Synthesize this post:\n" + json.dumps(payload, ensure_ascii=False)},
        ],
        options={"temperature": 0},
    )
    return PostUnderstanding.model_validate_json(response.message.content)


def extract_signals(model: str, post: dict, understanding: dict) -> SignalBundle:
    payload = {
        "post_id": str(post.get("post_id")),
        "text": post.get("text") or "",
        "understanding": understanding,
    }
    response = chat(
        model=model,
        format=SignalBundle.model_json_schema(),
        messages=[
            {"role": "system", "content": SYSTEM_SIGNAL},
            {"role": "user", "content": "Create narrow evidence signals for this post:\n" + json.dumps(payload, ensure_ascii=False)},
        ],
        options={"temperature": 0},
    )
    return SignalBundle.model_validate_json(response.message.content)


def main():
    ap = argparse.ArgumentParser(description="Analyze one enriched SelfTrace post with a local Ollama vision model")
    ap.add_argument("--input", default="data/processed/posts_enriched.json", help="Project-relative or absolute path")
    ap.add_argument("--post-id", default=None)
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--output", default=None, help="Project-relative or absolute output path")
    args = ap.parse_args()

    input_path = resolve_project_path(args.input)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input:        {input_path}")

    data = load_json(input_path)
    post = find_post(data, args.post_id)
    print(f"Analyzing post {post.get('post_id')} with {args.model}")

    observations = []
    for media in post.get("media", []):
        local_path = media.get("local_path")
        if not local_path or media.get("download_status") == "failed":
            observations.append({
                "media_index": media.get("index"),
                "status": "unavailable",
                "reason": media.get("download_error") or "missing local_path",
            })
            continue

        # IMPORTANT: local_path stored as data/media/... is resolved from PROJECT_ROOT.
        path = resolve_project_path(local_path)
        if not path.exists():
            observations.append({
                "media_index": media.get("index"),
                "status": "unavailable",
                "reason": f"file not found: {path}",
            })
            continue

        obs = analyze_media(args.model, path, post.get("text") or "")
        observations.append({"media_index": media.get("index"), "status": "ok", **obs.model_dump()})
        print(f"  media#{media.get('index')} -> {obs.media_type}")

    understanding = analyze_post_understanding(args.model, post, observations)
    signals = extract_signals(args.model, post, understanding.model_dump())

    output = {
        "schema_version": "selftrace-analysis-v0.2",
        "post_id": str(post.get("post_id")),
        "created_at": post.get("created_at"),
        "source_url": post.get("url"),
        "raw_text": post.get("text") or "",
        "media_observations": observations,
        "post_understanding": understanding.model_dump(),
        "evidence_signals": [s.model_dump() for s in signals.signals],
        "rejected_inferences": signals.rejected_inferences,
    }

    if args.output:
        out = resolve_project_path(args.output)
    else:
        out = (PROJECT_ROOT / "data" / "analysis" / f"{post.get('post_id')}.json").resolve()

    save_json(output, out)
    print(f"Saved: {out}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

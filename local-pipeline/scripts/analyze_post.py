#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_VERSION = "1.6"
SCHEMA_VERSION = "selftrace-analysis-v1.6"

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
        "AUTHOR_ORIGINAL", "COMMENTARY_ON_EXTERNAL_SOURCE", "SHARED_SOURCE_WITHOUT_CLEAR_COMMENTARY", "UNCLEAR"
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


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


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


def ollama_json(model: str, schema_model, system: str, user: str, images=None):
    message = {"role": "user", "content": user}
    if images:
        message["images"] = [str(x) for x in images]
    response = chat(
        model=model,
        format=schema_model.model_json_schema(),
        messages=[{"role": "system", "content": system}, message],
        options={"temperature": 0},
    )
    return schema_model.model_validate_json(response.message.content)


MEDIA_SYSTEM = """You are SelfTrace's objective visual-observation layer.
Return structured facts supported by the image only.
Rules:
- Do not infer personality or stable traits.
- Do not infer sensitive attributes.
- For text-heavy screenshots/notes, visible_text MUST contain representative legible text snippets and concepts MUST contain the main technical/semantic concepts.
- Do not put all information only in description.
- NOTE = text note / explanatory slide / bullet-note screenshot.
- PAPER_SCREENSHOT only when there is clear evidence of an academic paper page (paper title/authors/abstract/section/figure/citation formatting).
- TABLE only when content is genuinely organized in rows/columns.
- If text is unreadable, state that in uncertainty rather than inventing it.
"""

POST_SYSTEM = """You are SelfTrace's post-understanding layer.
Synthesize post text plus media observations.
Rules:
- primary_topics MUST contain 1-5 concrete topics when content is interpretable.
- post_types MUST contain at least one type.
- intents MUST contain at least one intent.
- explicit_statements MUST preserve the author's explicit claims/preferences/goals from the post text when present.
- Separate the author's own words from material merely shown in images.
- Do not infer stable personality or sensitive attributes.
"""

SIGNAL_SYSTEM = """You are SelfTrace's narrow evidence-signal layer.
Generate 0-4 useful signals from THIS POST ONLY.
Dimension semantics:
- INTEREST: explicit attraction, fascination, preference, curiosity toward a topic.
- KNOWLEDGE: demonstrated understanding or technical knowledge, not mere exposure.
- ACTIVITY: studying, reading, implementing, experimenting, documenting work.
- THINKING_PATTERN: a reasoning operation shown here (comparison, abstraction, analogy, mechanism-seeking, pattern-linking), phrased as behavior in this post only.
- EXPRESSION_STYLE: how the author communicates (concise, note-like, bilingual, question-driven), not what they are interested in.
- GOAL_ORIENTATION: explicit future intention, plan, learning/research goal.
Rules:
- HIGH = directly explicit in author text.
- MEDIUM = reasonable inference from behavior/content.
- Do NOT create a signal whose only claim is that the author attached/included an image showing X.
- Do NOT split one coherent idea into many near-duplicate signals.
- Do NOT infer stable personality, diagnosis, politics, religion, sexuality, health, ethnicity, or relationships.
"""


def media_needs_retry(obs: MediaObservation) -> bool:
    text_heavy = obs.media_type in {"SCREENSHOT", "PAPER_SCREENSHOT", "NOTE", "TABLE", "DIAGRAM", "CODE", "FORMULA", "OTHER"}
    return text_heavy and len(obs.description) > 80 and (not obs.visible_text or not obs.concepts)


def analyze_media(model: str, path: Path, post_text: str) -> MediaObservation:
    prompt = f"""Analyze this image attached to a social-media post.
Post text for context only:
{post_text or '[empty]'}

Populate every applicable field. For a text-heavy image, extract representative visible_text and concepts instead of leaving them empty."""
    obs = ollama_json(model, MediaObservation, MEDIA_SYSTEM, prompt, [path])

    if media_needs_retry(obs):
        retry = f"""Your previous structured output under-filled visible_text/concepts.
Previous output:
{json.dumps(obs.model_dump(), ensure_ascii=False)}

Re-analyze the SAME image. Correct media_type if needed. For any legible text-heavy image, visible_text and concepts must be populated with concise representative items."""
        obs = ollama_json(model, MediaObservation, MEDIA_SYSTEM, retry, [path])
    return obs


def post_needs_retry(u: PostUnderstanding, raw_text: str) -> bool:
    if not u.primary_topics or not u.post_types or not u.intents:
        return True
    if raw_text.strip() and not u.explicit_statements:
        return True
    return False


def analyze_post_understanding(model: str, post: dict, media_obs: list[dict]) -> PostUnderstanding:
    payload = {
        "post_id": str(post.get("post_id")),
        "created_at": post.get("created_at"),
        "text": post.get("text") or "",
        "media_observations": media_obs,
    }
    user = "Synthesize this post into the required structured fields:\n" + json.dumps(payload, ensure_ascii=False)
    u = ollama_json(model, PostUnderstanding, POST_SYSTEM, user)

    if post_needs_retry(u, payload["text"]):
        retry = f"""The previous output was incomplete.
Previous output:
{json.dumps(u.model_dump(), ensure_ascii=False)}

Return a corrected output. primary_topics, post_types, and intents must be non-empty for interpretable content. Because the raw post text is non-empty, explicit_statements must capture the author's explicit statements/preferences/goals."""
        u = ollama_json(model, PostUnderstanding, POST_SYSTEM, retry)
    return u


def is_media_only_signal(sig: EvidenceSignal) -> bool:
    t = (sig.claim + " " + " ".join(sig.evidence)).lower()
    patterns = [
        "included media", "attached media", "attached image", "included image",
        "media showing", "image showing", "screenshot showing"
    ]
    return any(p in t for p in patterns)


def semantic_dimension(sig: EvidenceSignal) -> str:
    t = (sig.claim + " " + " ".join(sig.evidence)).lower()
    if re.search(r"fascinat|interest|interested|curios|drawn to|appeal", t):
        return "INTEREST"
    if re.search(r"would like|plan|intend|future|goal|systematically study|want to study|aim", t):
        return "GOAL_ORIENTATION"
    if re.search(r"implement|experiment|studying|study of|learning|reading|documenting|working on|engag", t):
        return "ACTIVITY"
    if re.search(r"compare|comparison|abstract|analogy|same core|shared core|mechanism|pattern|connects .* concepts", t):
        return "THINKING_PATTERN"
    if re.search(r"demonstrates understanding|understands|explains|technical knowledge|knowledge of", t):
        return "KNOWLEDGE"
    return sig.dimension


def normalize_signals(bundle: SignalBundle) -> SignalBundle:
    kept = []
    seen = set()
    rejected = list(bundle.rejected_inferences)

    for sig in bundle.signals:
        if is_media_only_signal(sig):
            rejected.append(f"Filtered non-signal media-description claim: {sig.claim}")
            continue

        sig.dimension = semantic_dimension(sig)  # pydantic model is mutable
        key = (sig.dimension, re.sub(r"\W+", " ", sig.claim.lower()).strip())
        if key in seen:
            continue
        seen.add(key)
        kept.append(sig)

    # Keep at most 4 strongest/useful signals; HIGH first.
    kept.sort(key=lambda s: 0 if s.confidence == "HIGH" else 1)
    return SignalBundle(signals=kept[:4], rejected_inferences=rejected)


def extract_signals(model: str, post: dict, understanding: PostUnderstanding) -> SignalBundle:
    payload = {
        "post_id": str(post.get("post_id")),
        "raw_text": post.get("text") or "",
        "post_understanding": understanding.model_dump(),
    }
    bundle = ollama_json(
        model,
        SignalBundle,
        SIGNAL_SYSTEM,
        "Create narrow evidence signals for this post:\n" + json.dumps(payload, ensure_ascii=False),
    )
    return normalize_signals(bundle)


def main():
    ap = argparse.ArgumentParser(description="SelfTrace single-post multimodal analyzer v1.6")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--post-id", default=None)
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    input_path = resolve_project_path(args.input)
    data = load_json(input_path)
    post = find_post(data, args.post_id)

    print(f"SelfTrace analyzer v{ANALYSIS_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input: {input_path}")
    print(f"Analyzing post {post.get('post_id')} with {args.model}")

    observations = []
    for media in post.get("media", []):
        local_path = media.get("local_path")
        if not local_path or media.get("download_status") == "failed":
            observations.append({
                "media_index": media.get("index"), "status": "unavailable",
                "reason": media.get("download_error") or "missing local_path",
            })
            continue
        path = resolve_project_path(local_path)
        if not path.exists():
            observations.append({
                "media_index": media.get("index"), "status": "unavailable",
                "reason": f"file not found: {path}",
            })
            continue
        obs = analyze_media(args.model, path, post.get("text") or "")
        observations.append({"media_index": media.get("index"), "status": "ok", **obs.model_dump()})
        print(f"  media#{media.get('index')} -> {obs.media_type}, concepts={len(obs.concepts)}, visible_text={len(obs.visible_text)}")

    understanding = analyze_post_understanding(args.model, post, observations)
    signals = extract_signals(args.model, post, understanding)

    output = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "model": args.model,
        "post_id": str(post.get("post_id")),
        "created_at": post.get("created_at"),
        "source_url": post.get("url"),
        "raw_text": post.get("text") or "",
        "media_observations": observations,
        "post_understanding": understanding.model_dump(),
        "evidence_signals": [s.model_dump() for s in signals.signals],
        "rejected_inferences": signals.rejected_inferences,
    }

    out = resolve_project_path(args.output) if args.output else (PROJECT_ROOT / "data" / "analysis" / f"{post.get('post_id')}.json").resolve()
    save_json(output, out)
    print(f"Signals kept: {len(signals.signals)}")
    for s in signals.signals:
        print(f"  {s.dimension}/{s.confidence}: {s.claim}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

# SelfTrace

**Evidence-based longitudinal personal modeling from digital traces.**

SelfTrace turns personal digital traces into a structured, time-aware, and traceable **Personal Model**.

```text
Digital Trace
    ↓
Evidence
    ↓
Temporal Pattern
    ↓
Personal Model
    ↓
Evidence Presentation
```

## 1. Goal

SelfTrace is designed to answer three questions:

- **What changed?**
- **What persisted?**
- **What is emerging?**

The goal is not to infer personality from a single post, but to build a longitudinal model from repeated, evidence-backed observations.

## 2. V1 Scope

Current V1 uses the user's own **X posts** as the data source.

Supported content:

- Text
- Images
- Timestamps
- Source URLs

Current Personal Model dimensions:

- Interest
- Knowledge
- Activity
- Goal Orientation
- Thinking Pattern
- Expression Style

## 3. Pipeline

### Stage 1 — Data Collection

Collect the user's own X posts through a Chrome extension.

Output:

```text
X posts JSON
```

### Stage 2 — Cleaning & Media Localization

- Deduplicate posts
- Validate data
- Download media locally
- Build enriched post records

Output:

```text
data/processed/posts_enriched.json
data/processed/media_manifest.json
data/media/
```

### Stage 3 — Per-Post Multimodal Analysis

Analyze each post with:

```text
Text + Images
    ↓
Media Observations
    ↓
Post Understanding
    ↓
Evidence Signals
```

Each signal contains:

- Dimension
- Claim
- Evidence
- Confidence
- Source post

### Stage 4 — Batch Analysis

Run Stage 3 over all posts.

Providers:

- `openai`
- `local` via Ollama

Output:

```text
data/analysis/openai/
data/analysis/local/
```

### Stage 5 — Temporal Aggregation

Aggregate post-level signals across time.

Computes:

- Frequency
- First / last seen
- Active span
- Recency
- Monthly counts
- Temporal status
- Evidence provenance

Output:

```text
data/temporal/temporal_aggregation.json
data/temporal/temporal_report.md
```

### Stage 6 — Personal Model Generation

Generate higher-level, evidence-backed Personal Model claims.

Model structure:

```text
Personal Model
├─ Cognitive Model
├─ Motivational Model
├─ Behavioral Model
├─ Communicative Model
├─ Current / Emerging State
└─ Evolution Patterns
```

Output:

```text
data/personal/personal_model.json
data/personal/personal_model.md
```

### Stage 7 — Evidence Presentation

Render the Personal Model into a local HTML page.

Each high-level claim can be traced back to:

```text
Personal Model Claim
    ↓
Evidence Signal
    ↓
Supporting Post
    ↓
Original Evidence
    ↓
Source URL
```

Output:

```text
data/presentation/personal_model.html
```

## 4. Run

Activate the environment:

```bash
cd ~/Documents/project/mine_ai_project/self_trace/local-pipeline
source .venv/bin/activate
```

### Analyze all posts

```bash
python scripts/analyze_all_posts.py \
  --input data/processed/posts_enriched.json \
  --provider openai
```

### Temporal aggregation

```bash
python scripts/aggregate_temporal.py \
  --input-dir data/analysis/openai \
  --provider openai
```

### Generate Personal Model

```bash
python scripts/generate_personal_model.py \
  --input data/temporal/temporal_aggregation.json \
  --provider openai
```

### Render HTML

```bash
python scripts/render_personal_model.py \
  --input data/personal/personal_model.json
```

Then open:

```text
data/presentation/personal_model.html
```

## 5. Design Principles

### Evidence first

No direct:

```text
Post → Personality
```

Instead:

```text
Post
→ Evidence Signal
→ Temporal Pattern
→ Personal Model Claim
```

### Longitudinal, not static

A Personal Model should capture:

- Emerging patterns
- Recurring patterns
- Current focus
- Historical changes

### Traceable

Every high-level claim should preserve its supporting evidence and source.

### User-controlled

SelfTrace is designed around user-owned data and local processing where possible.

## 6. Current Status

**SelfTrace V1 is end-to-end complete.**

```text
Collection
→ Cleaning
→ Multimodal Analysis
→ Batch Analysis
→ Temporal Aggregation
→ Personal Model
→ Evidence Presentation
```

The current implementation has successfully produced a traceable Personal Model from the collected X dataset.

## 7. Boundaries

SelfTrace models **observed digital traces**, not the whole person.

It should not treat:

- Missing evidence as absence of a trait or interest
- Single observations as stable characteristics
- Generated claims as psychological diagnosis
- Sparse historical data as a complete life history

## 8. Future Direction

Possible future extensions:

```text
X
GitHub
Notes
Papers
Projects
Photos
Chat
Calendar
    ↓
Unified Personal Model
```

The long-term direction is a user-owned, evidence-based personal context layer for memory, reflection, search, personalization, and personal AI agents.

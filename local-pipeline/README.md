# SelfTrace Analysis v2.0 — OpenAI + Local switch

Incremental upgrade for your existing project:

```text
self_trace/
└── local-pipeline/
    ├── config.json
    ├── .env.local              # your key stays on your Mac; not included here
    ├── requirements.txt
    ├── scripts/
    │   ├── analyze_post.py
    │   ├── analyze_all_posts.py
    │   └── signal_validator.py
    ├── data/
    └── logs/
```

Copy the files from this package into your existing `local-pipeline/`. **Do not replace your existing `data/` folder.**

## What changed

### Two providers, one pipeline

`config.json` defaults to the cheapest current GPT-5.6 hosted model:

```json
{
  "provider": "openai",
  "openai": {
    "model": "gpt-5.6-luna",
    "image_detail": "high",
    "retries": 1
  },
  "local": {
    "model": "qwen3-vl:8b",
    "retries": 1
  }
}
```

Switch permanently by changing only:

```json
"provider": "openai"
```

to:

```json
"provider": "local"
```

Or override for one command without touching the config:

```bash
--provider openai
```

or:

```bash
--provider local
```

### One post = one model call

Both providers now receive:

```text
post text + all post images
        ↓
one multimodal model call
        ↓
media_observations
post_understanding
candidate evidence_signals
        ↓
Python signal validator
```

This replaces the older per-image + synthesis + signal multi-call path.

## API key

The OpenAI backend reads `OPENAI_API_KEY` from either:

- your shell environment, or
- `local-pipeline/.env.local`.

The program never prints the key.

A `.env.example` template is included. Keep `.env.local` only on your Mac and do not commit/share it.

## Install/update dependencies

From your existing `local-pipeline/` directory:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## First hosted smoke test: one post

```bash
python scripts/analyze_post.py \
  --input data/processed/posts_enriched.json \
  --post-id 2092794113787646099 \
  --provider openai
```

The output goes to:

```text
data/analysis/openai/2092794113787646099.json
```

## Compare with local using the same code

```bash
python scripts/analyze_post.py \
  --input data/processed/posts_enriched.json \
  --post-id 2092794113787646099 \
  --provider local
```

The local result goes to:

```text
data/analysis/local/2092794113787646099.json
```

So OpenAI and local results do not overwrite each other.

## 3-post hosted smoke test

```bash
python scripts/analyze_all_posts.py \
  --input data/processed/posts_enriched.json \
  --provider openai \
  --limit 3
```

## Full hosted run

```bash
python scripts/analyze_all_posts.py \
  --input data/processed/posts_enriched.json \
  --provider openai
```

Resume is enabled by default. Existing successful v2.0 outputs for the same provider/model are skipped.

## Switch full batch back to local

```bash
python scripts/analyze_all_posts.py \
  --input data/processed/posts_enriched.json \
  --provider local
```

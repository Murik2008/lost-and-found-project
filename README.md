# Smart Lost & Found

Match lost items against found items using a vision-language description plus embedding similarity.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `gemini` | VLM used by `ai.describe_item` |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `openai` | Embedder used by `ai.embed` |
| `DATABASE_URL` | empty (in-memory) | Set to Postgres URL for persistence |
| `OFFLINE_MODE` | `false` | `true` = optional deterministic fake AI (dev only) |
| `DATA_DIR` | `data` | Image blob storage |
| `LOG_LEVEL`, `MAX_RETRIES`, `CONCURRENCY_LIMIT` | `INFO`, `3`, `8` | Logging, retries, parallel AI calls |

## Run

```bash
# API + minimal web UI at http://localhost:8000/
uvicorn src.api:app --host 0.0.0.0 --port 8000

# CLI
python -m src.cli register-lost -d "black umbrella" -i data/lost/umbrella_black.png
python -m src.cli register-found -d "black umbrella found" -i data/found/umbrella_black_2.png
python -m src.cli list --type lost
python -m src.cli search-matches --item-id <id> -k 3
```

Web UI: open `/` in a browser. API docs: `/docs`.

## Test

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

No test hits the network. The provided `ai/` smoke flow can be checked offline:

```bash
python sendedfolder/topic-1-lost-and-found/demo_ai.py --offline
```

## Docker

```bash
cp .env.example .env
docker compose up --build
# app at http://localhost:8000/
```

With Postgres:

```bash
docker compose up --build
# DATABASE_URL=postgresql://lostfound:lostfound@db:5432/lostfound
```

Without compose (in-memory store):

```bash
docker build -t lostfound .
docker run -p 8000:8000 lostfound
```

## API

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/items"              # all
curl "http://localhost:8000/items?type=lost"    # lost only
curl "http://localhost:8000/items?type=found"   # found only
curl "http://localhost:8000/items?status=pending"
curl -F "image=@data/lost/umbrella_black.png" -F "user_description=black umbrella" \
  http://localhost:8000/items/lost
curl "http://localhost:8000/items/<id>"
curl "http://localhost:8000/items/<id>/image"   # photo
curl "http://localhost:8000/items/<id>/matches?k=5&min_score=0.5"
curl -X POST -H "Content-Type: application/json" \
  -d '{"other_item_id":"<other-id>"}' http://localhost:8000/items/<id>/match
curl "http://localhost:8000/items/<id>/pair"   # what it was matched with
curl -X PATCH -H "Content-Type: application/json" -H "X-Owner-Token: <token>" \
  -d '{"status":"matched"}' http://localhost:8000/items/<id>/status
curl -X DELETE -H "X-Owner-Token: <token>" http://localhost:8000/items/<id>
```

Ownership: browsing, photos and matches are public on the site. The site has
no edit or delete buttons by design — status changes and deletes are done
through the API (Postman collection included) or the CLI.

Import `postman_collection.json` into Postman for one-click requests
(set `base_url`, then copy an id into `item_id`).

## How matching works

1. **Describe** — the vision model reads the photo + your text and writes a
   structured note (object, colors, brand, marks).
2. **Embed** — the note becomes a vector; similar meanings give similar vectors.
3. **Rank** — cosine similarity against the opposite pool, best first, with a
   reason line (shared object / color / brand). The score ranks candidates;
   it is not a percentage.

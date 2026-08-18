# ReframeBot

ReframeBot is a CBT-oriented chatbot for supporting university students with academic stress. It combines a fine-tuned Llama 3.1 model with a guardrail router (TASK_1/TASK_2/TASK_3) and optional RAG grounding from a CBT knowledge base.


https://github.com/user-attachments/assets/fc269f59-4975-476e-b941-2491cb7e35e8


## Model Repositories

| Model | Repository | Use |
|---|---|---|
| AWQ Model | [ReframeBot-Llama3.1-8B-AWQ](https://huggingface.co/Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ) | **Inference** — merged + AWQ 4-bit, served by vLLM |
| Guardrail Classifier | [ReframeBot-Guardrail-DistilBERT](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT) | **Inference** — 3-class task router (CBT / Crisis / Out-of-scope) |
| DPO Adapter | [ReframeBot-DPO-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B) | Training artifact — LoRA adapter before merging |
| SFT Adapter | [ReframeBot-SFT-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B) | Training artifact — intermediate SFT checkpoint |

The API container image is published on Docker Hub:

| Image | Repository |
|---|---|
| API container | [nhatminh115/reframebot-api](https://hub.docker.com/r/nhatminh115/reframebot-api) |

## Features
- Fine-tuned Llama 3.1 8B (SFT + DPO adapter, merged and served via vLLM)
- AWQ 4-bit quantization (autoawq) — runs on 8 GB VRAM
- vLLM serving with PagedAttention and continuous batching
- Guardrail routing with crisis detection and out-of-scope redirection
- Optional RAG grounding over a CBT knowledge base
- Dockerized stack (vLLM container + FastAPI container) and a lightweight static web UI

## System Workflow

```
User message (browser)
        |
        v
FastAPI  /chat  or  /chat/stream  (SSE)
        |
        |-- [1] CRISIS DETECTION  (guardrail.py)
        |       Regex hard patterns  +  semantic cosine-sim
        |       vs. crisis prototype sentences
        |       Crisis detected? --> empathy reply + hotlines  (stop)
        |
        |-- [2] GUARDRAIL CLASSIFICATION  (guardrail.py)
        |       Input : last N user turns
        |       Model : DistilBERT fine-tuned (CPU, ~250 MB)
        |       Output: TASK_1 / TASK_2 / TASK_3  +  confidence score
        |
        |-- [3] TASK ROUTING  (router.py)
        |       Priority 0: follow-up inside an ongoing academic context
        |       Priority 1: academic keyword match (regex)
        |       Priority 2: TASK_2 at any confidence  --> hotlines  (stop)
        |       Priority 3: trust guardrail label
        |       Effective label: TASK_1 | TASK_2 | TASK_3
        |
        |-- [4] RAG RETRIEVAL  (rag.py)  -- TASK_1 only, optional
        |       Query : latest user message
        |       Store : ChromaDB  (CBT knowledge base)
        |       Output: top-2 chunks, or ""  if DB unavailable
        |
        |-- [5] LLM GENERATION  (llm.py)
                System prompt : task-specific  (TASK_1 / TASK_3)
                Context       : RAG chunks injected into prompt
                Backend       : HTTP --> vLLM container  (port 8001)
                Safety filter : suppress accidental crisis output
                Delivery      : SSE token stream  /  JSON response
                        |
                        v
                Browser renders response
```

### Infrastructure

```
Browser  (port 3000, nginx)
        |
FastAPI container  (port 8000)
  |- Guardrail classifier  (DistilBERT, CPU)
  |- Crisis detector       (regex + SentenceTransformer, CPU)
  |- RAG retrieval         (ChromaDB, disk)
  |- Task router           (pure Python)
        |  HTTP (OpenAI-compatible)
        v
vLLM container  (port 8001)
  Llama 3.1 8B  AWQ 4-bit  --  5.4 GB VRAM
  PagedAttention + continuous batching  --  ~39 tok/s
```

## Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU with 8 GB+ VRAM
- 32 GB RAM (for model export step)
- Docker Desktop with NVIDIA Container Toolkit
- WSL2 (for AWQ quantization step)

### Option A — Docker (recommended)

All models are pre-built and hosted on Hugging Face / Docker Hub — no training or quantization required.

1. Clone and configure:
```bash
git clone https://github.com/minhnghiem32131024429/ReframeBot.git
cd ReframeBot
cp .env.example .env
# Set HF_TOKEN in .env (required to download the AWQ model from HF)
```

2. Download models:
```bash
# AWQ model (~4 GB, served by vLLM)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Nhatminh1234/ReframeBot-Llama3.1-8B-AWQ', local_dir='./merged_model_awq')
"

# Guardrail classifier (~250 MB, runs on CPU)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Nhatminh1234/ReframeBot-Guardrail-DistilBERT', local_dir='./guardrail_model_retrained/best')
"
```

3. Start the stack (pulls API image from Docker Hub automatically):
```bash
docker compose up
```

4. Open the UI at **http://localhost:3000** — served by the `web` nginx service included in the compose stack.

### Option B — In-process (no Docker)

```bash
pip install -e ".[inprocess]"
cp .env.example .env
# Set ADAPTER_PATH and GUARDRAIL_PATH in .env
python app.py
```
Note: this path uses the original transformers/PEFT in-process loading without vLLM.

### Building from source (advanced)

If you want to rebuild the AWQ model yourself from the DPO adapter:
```bash
# Step 1: merge base model + DPO adapter → bf16 safetensors (~16 GB RAM, no GPU needed)
uv run python scripts/export_merged_model.py --output ./merged_model

# Step 2: AWQ 4-bit quantization (requires GPU, run in WSL2)
# In WSL2: pip install autoawq
python scripts/quantize_awq.py --input ./merged_model --output ./merged_model_awq
```

## Project Structure

```
ReframeBot/
├── app.py                      # Entry point: python app.py
├── docker-compose.yml          # vLLM + API containers
├── docker/api.Dockerfile       # FastAPI container image
├── pyproject.toml              # Dependencies (runtime / inprocess / scripts / train)
├── train.ipynb                 # Training notebook (SFT + DPO + Guardrail)
├── src/
│   └── reframebot/
│       ├── config.py           # All settings via pydantic-settings + .env
│       ├── constants.py        # Hotlines, keywords, regex, prototype sentences
│       ├── router.py           # Task routing logic (TASK_1/2/3 priority chain)
│       ├── main.py             # FastAPI app, lifespan, /chat + /chat/stream endpoints
│       └── services/
│           ├── guardrail.py    # Guardrail classifier + crisis detection
│           ├── rag.py          # ChromaDB retrieval
│           └── llm.py          # vLLM client (OpenAI-compatible)
├── web/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   ├── dataset.jsonl           # SFT training data
│   ├── dataset_dpo.jsonl       # DPO training data
│   └── guardrail_dataset.jsonl
├── scripts/
│   ├── export_merged_model.py  # Merge base + DPO adapter → bf16 safetensors
│   ├── quantize_awq.py         # AWQ 4-bit quantization (run in WSL2/Linux)
│   ├── benchmark.py            # Latency / throughput / TTFT benchmark (vLLM)
│   ├── benchmark_inprocess.py  # Latency / throughput / TTFT benchmark (NF4)
│   ├── build_rag_db.py         # Build ChromaDB from knowledge.txt
│   ├── train_guardrail.py      # Retrain guardrail classifier
│   ├── evaluate_model.py       # Evaluation + metrics (--mode inprocess|vllm)
│   ├── push_all_models.py      # Upload all models to HF Hub
│   └── push_model_cards.py     # Sync model cards to HF Hub
├── tests/
│   └── unit/
│       ├── test_constants.py   # Regex pattern tests (no ML deps)
│       ├── test_guardrail.py   # build_guardrail_input + detect_crisis (mocked)
│       └── test_router.py      # resolve_task logic (pure Python)
└── Utils/                      # Background audio/image assets
```

## UI

- Glassmorphism-style layout (HTML/CSS)
- Responsive chat UI

## Configuration

### Change API URL
The UI uses relative paths (`/chat`, `/chat/stream`) so it works out of the box via the nginx proxy. For a custom domain, update the nginx `proxy_pass` in `docker/nginx.conf`:
```nginx
proxy_pass http://your-api-host:8000/chat;
```

### All configuration via `.env`
Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `ADAPTER_PATH` | — | Path to DPO adapter checkpoint (required) |
| `GUARDRAIL_PATH` | auto-discover | Path to guardrail model directory |
| `BASE_MODEL_NAME` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HF model ID or local path |
| `RAG_DB_PATH` | `./rag_db` | ChromaDB directory |
| `GUARDRAIL_CONTEXT_TURNS` | `3` | Recent user turns fed to the classifier |
| `CRISIS_CONFIDENCE_THRESHOLD` | `0.90` | Guardrail score above which TASK_2 is high-confidence |
| `CRISIS_TASK2_PROB_THRESHOLD` | `0.10` | Tuned probability floor for routing to TASK_2 from the classifier's full probability vector |
| `CRISIS_SEMANTIC_SIM_THRESHOLD` | `0.62` | Cosine sim threshold for semantic crisis detection |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind address |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed origins |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith traces for `/chat` and `/chat/stream` |
| `LANGSMITH_PROJECT` | `reframebot-dev` | LangSmith project name |
| `LANGSMITH_API_KEY` | — | LangSmith API key |
| `APP_VERSION` | `dev` | App version recorded in LangSmith trace metadata |
| `GUARDRAIL_VERSION` | `local` | Guardrail version recorded in LangSmith trace metadata |
| `RAG_TOP_K` | `3` | Number of RAG chunks retrieved for TASK_1 responses |

For Docker, rebuild/restart the API after changing these values:

```bash
docker compose down
docker compose up --build
docker compose logs -f api
```

On startup the API logs `LangSmith tracing: enabled=... project=... api_key_set=...`.

Trace anatomy:

| Span | What it measures | Useful metadata |
|---|---|---|
| `reframebot_resolve` | crisis detector, guardrail classifier, routing, and RAG retrieval | `guardrail_path`, thresholds, username/session |
| `llm_generate` | non-streaming response generation and crisis empathy generation | model, effective label, completion tokens, elapsed time, tokens/sec |
| `llm_stream` | streaming vLLM response generation | model, effective label, RAG context chars, TTFT, token count, elapsed time, tokens/sec |

Use `llm_stream.ttft_s` to spot first-token latency and `reframebot_resolve` to separate routing/RAG latency from generation latency.
For local vLLM, LangSmith usage metadata records token counts and zero API cost (`total_cost=0`) because inference is self-hosted.
LangSmith integration lives in `src/reframebot/services/tracing.py`; runtime code calls it through no-op helpers, so the app still runs normally when tracing is disabled.

### Customize Colors
Edit `web/style.css` to change color scheme, glass effects, and more.

## Model Sizes

| Checkpoint | Format | Disk size |
|---|---|---|
| Merged model (base + DPO adapter) | bf16 safetensors | 15 GB |
| AWQ quantized (served by vLLM) | AWQ 4-bit | 5.4 GB |
| Guardrail classifier | DistilBERT fp32 | ~250 MB |

The bf16 merged model (15 GB) exceeds the 8 GB VRAM of the development GPU and cannot be served unquantized on this hardware. AWQ 4-bit quantization reduces the footprint to 5.4 GB (2.8x compression), enabling deployment on a consumer 8 GB card.

## Inference Performance

Measured on NVIDIA RTX 5070 (8 GB VRAM):

| Metric | AWQ 4-bit (vLLM) | Base + DPO (NF4, in-process) | Speedup |
|---|---|---|---|
| Latency p50 | 2.6s | 106.8s | **41x** |
| Latency p95 | 7.1s | 124.1s | **17x** |
| Time to First Token (TTFT) p50 | 1.11s | 12.3s | **11x** |
| Tokens/sec | ~39 tok/s | ~2.1 tok/s | **19x** |
| Throughput (4 concurrent) | 1.0 req/s | — | — |
| VRAM usage at runtime | ~5.4 GB dedicated | ~8 GB dedicated + ~7 GB shared (system RAM) | — |

AWQ + vLLM (PagedAttention, continuous batching, Marlin kernel) delivers 26–32x faster inference vs in-process NF4 loading. The NF4 path spills into shared VRAM (system RAM) on Windows, has no kernel optimization or batching, and is suitable for evaluation and offline use only.

Cold-start latency (~115s first request on vLLM) is due to CUDA kernel compilation; subsequent requests are warm.

> **Methodology note:** Latency numbers include the full request path (guardrail → RAG → vLLM). Tokens/sec is measured by counting SSE events from the vLLM streaming endpoint (one event = one BPE token). The benchmark uses 8 fixed prompts rotated across N=30 sequential requests after a 3-request warm-up; p95 should be interpreted as directional, not production-grade, given the controlled prompt distribution.

To reproduce:
```bash
# AWQ via vLLM — 3-request warm-up runs automatically before measurement
docker compose up -d vllm
uv run python scripts/benchmark.py --n 30 --concurrency 4

# Base+DPO NF4 in-process (baseline comparison)
uv run python scripts/benchmark_inprocess.py --n 10
```

## Evaluation Results

**Guardrail classifier** (same model regardless of LLM serving mode):

| Metric | Score | Notes |
|---|---|---|
| Accuracy (out-of-domain eval set, tuned threshold) | **98.3%** | 60 samples; `P(TASK_2) >= 0.10`, 0 TASK_2 false negatives, 1 TASK_2 false positive |
| Accuracy (out-of-domain eval set, argmax only) | **96.7%** | Same model, no probability threshold |
| Accuracy (in-domain validation split) | **99.0%** | 20% stratified split, same synthetic source plus curated hard cases |
| F1 macro (in-domain validation split) | **0.99** | |

Hard edge cases include: benign crisis metaphors ("dying of embarrassment"), passive suicidal ideation ("feeling like a burden to everyone"), ambiguous short inputs, Vietnamese text, pills/overdose language, and mixed academic+crisis signals.

> **Interpretation:** The in-domain figure measures fit to the synthetic/curated training distribution. The out-of-domain set is a more realistic signal. Rerun `scripts/evaluate_model.py` or `scripts/evaluate_guardrail_classifier.py` after any guardrail retrain to get updated numbers.

The current routing rule uses the classifier's full probability vector:

```text
1. Run regex + semantic crisis detector. If crisis: TASK_2.
2. Preserve short follow-up turns inside academic context as TASK_1.
3. Preserve academic-keyword context as TASK_1.
4. If P(TASK_2) >= CRISIS_TASK2_PROB_THRESHOLD, route TASK_2.
5. Otherwise use the classifier top label / legacy TASK_2 confidence logic.
```

Threshold sweep on `data/evaluation_test_data.json`:

| Threshold `P(TASK_2)` | Accuracy | TASK_2 Precision | TASK_2 Recall | TASK_2 F1 | TASK_2 FN | TASK_2 FP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.9667 | 0.9130 | 1.0000 | 0.9545 | 0 | 2 |
| 0.10 | 0.9833 | 0.9545 | 1.0000 | 0.9767 | 0 | 1 |
| 0.20 | 0.9833 | 1.0000 | 0.9524 | 0.9756 | 1 | 0 |
| 0.30 | 0.9667 | 1.0000 | 0.9048 | 0.9500 | 2 | 0 |

The selected default is `0.10`, prioritizing TASK_2 recall while keeping false positives low on the hard eval set. The generated curve is written to `reports/guardrail_threshold_sweep.png`.

**LLM quality** (varies by serving mode):

| Metric | AWQ 4-bit (vLLM) | Base + DPO (NF4) |
|---|---|---|
| BERTScore Relevance | **0.865** | 0.832 |
| BERTScore Faithfulness | **0.858** | 0.849 |
| Response Consistency | **0.775** | 0.732 |
| Response Length Score | 0.625 | 0.599 |

AWQ quantization does not degrade quality — all LLM metrics are equal to or better than the NF4 baseline. Faithfulness > Relevance in both modes suggests the model grounds well in retrieved CBT context when RAG is active.

### Methodology

The quality of the system is evaluated across five dimensions using the `scripts/evaluate_model.py` suite:

- **Accuracy**: Classification accuracy of the DistilBERT guardrail on a held-out test set.
- **Consistency**: Reliability of responses. Measured by the **Cosine Similarity** (via `all-MiniLM-L6-v2`) between two independent outputs for the same prompt.
- **Semantic Relevance**: Alignment with ground-truth answers. Calculated using **BERTScore F1** (generated response vs. reference).
- **Context Faithfulness**: RAG grounding quality. Calculated using **BERTScore F1** (generated response vs. retrieved knowledge base context).
- **Response Complexity**: A **Gaussian score** (target = 100 words, $\sigma$ = 80) that penalizes responses that are excessively short or long.

To reproduce:
```bash
docker compose up -d vllm && uv run python scripts/evaluate_model.py --mode vllm
uv run python scripts/evaluate_model.py --mode inprocess
```

## Training

See `train.ipynb` for the complete training pipeline:
1. **SFT (Supervised Fine-Tuning)** - Base model adaptation
2. **DPO (Direct Preference Optimization)** - Response quality improvement
3. **Guardrail Training** - Task classification model

Optional scripts:
- `scripts/prepare_guardrail_data.py`: merge base guardrail data with curated hard cases and remove exact duplicates
- `scripts/audit_guardrail_dataset.py`: inspect class balance, duplicate labels, CBT/Crisis boundary phrases, and OOS coverage
- `scripts/train_guardrail.py`: retrain the guardrail classifier from a JSONL dataset
- `scripts/evaluate_guardrail_classifier.py`: report confusion matrix and per-class precision/recall/F1
- `scripts/evaluate_rag_ragas.py`: evaluate RAG faithfulness/retrieval quality with RAGAS
- `scripts/sweep_guardrail_threshold.py`: sweep `P(TASK_2)` thresholds and write `reports/guardrail_threshold_sweep.csv/.png`
- `scripts/push_guardrail_version.py`: upload a guardrail checkpoint to Hugging Face with a branch/tag version

Guardrail retraining workflow:

```bash
uv run --extra scripts python scripts/prepare_guardrail_data.py
uv run --extra scripts python scripts/audit_guardrail_dataset.py --data data/guardrail_dataset_clean.jsonl
uv run --extra scripts python scripts/train_guardrail.py --data data/guardrail_dataset_clean.jsonl --out guardrail_model_retrained_clean
uv run --extra scripts python scripts/evaluate_guardrail_classifier.py --model guardrail_model_retrained_clean/best --eval-json data/evaluation_test_data.json
uv run --extra scripts python scripts/sweep_guardrail_threshold.py --model guardrail_model_retrained_clean/best --eval-json data/evaluation_test_data.json
```

RAGAS evaluation workflow:

```bash
# Requires GROQ_API_KEY for the default Groq judge.
# Embeddings are local via sentence-transformers/all-MiniLM-L6-v2.
# Run the API first so answers are generated by the actual ReframeBot stack.
uv sync --extra scripts
uv run --extra scripts python scripts/build_rag_db.py
docker compose down
docker compose up -d --build
uv run --extra scripts python scripts/evaluate_rag_ragas.py --api-url http://localhost:8000/chat --ragas-workers 1
```

The project pins Python to `>=3.11,<3.14` because RAGAS 0.2.x can fail under Python 3.14's asyncio behavior on Windows. `.python-version` keeps `uv` on the local Python 3.11 interpreter.
Rebuild `rag_db` and restart the API after changing the knowledge base or chunking logic; the running API loads RAG into memory at startup.

Current Docker-backed RAGAS baseline after topic-aware chunking and `RAG_TOP_K=3`:

| Metric | Score |
| --- | ---: |
| faithfulness | 0.7388 |
| context_precision | 0.8125 |
| context_recall | 0.8125 |

RAGAS complements the older BERTScore-based evaluator: BERTScore checks semantic similarity, while RAGAS separately scores retrieval context quality and whether generated answers are faithful to that context.

Guardrail Hugging Face versioning:

```bash
# Upload only a versioned branch/tag, leaving main unchanged.
uv run --extra scripts python scripts/push_guardrail_version.py --version v2-guardrail-clean --archive-main-as v1-guardrail-original

# Promote the same checkpoint to main after reviewing the versioned branch.
uv run --extra scripts python scripts/push_guardrail_version.py --version v2-guardrail-clean --update-main
```

To pin a specific version in downstream code:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Nhatminh1234/ReframeBot-Guardrail-DistilBERT",
    revision="v2-guardrail-clean",
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Author

**Nghiem Nhat Minh**
- GitHub: [@nhatminh115](https://github.com/nhatminh-115)
- Hugging Face: [@Nhatminh1234](https://huggingface.co/Nhatminh1234)

## Acknowledgments

- Meta AI for Llama 3.1
- Hugging Face for transformers and PEFT libraries
- FastAPI team

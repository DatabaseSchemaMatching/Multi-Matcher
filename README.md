# Multi-Matcher: Multi-Model Schema Matching via Unified Schema and Adaptive Filtering with Pre-Trained Language Models

Multi-Matcher is a multi-model schema matching pipeline that:

1. Builds schema contexts (table / document / graph)
2. Retrieves candidates via embedding cosine similarity (ChromaDB)
3. Filters candidates with adaptive thresholding (Kneedle)
4. Groups schema elements using an LLM

## Repo Structure

- `scripts/run_dataset.py`: CLI runner
- `src/multimatcher/`: core library
- `.env.example`: environment variable template

## Setup

### Clone the Repository

```bash
git clone https://github.com/DatabaseSchemaMatching/Multi-Matcher.git
cd Multi-Matcher
```

### 1) Create virtual environment (recommended)

```bash
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### Data Preparation

Download the dataset archive from the following link and extract it to a local directory:

- Dataset: https://drive.google.com/file/d/1ORXPF3W3mASLdetCnXtFevewg_hjl2jQ/view?usp=sharing

After extracting, set `MULTIMATCHER_DATA_ROOT` (in `.env`) to the extracted dataset root directory so that it contains:

- `M2Bench_Ecommerce/`
- `M2Bench_Healthcare/`
- `Unibench/`
- `M2E_Unibench/`

### 3) Configure environment variables

Copy `.env.example` to `.env` and fill your keys.

```bash
# Windows (PowerShell)
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Open `.env` and set at least the following:

- `MULTIMATCHER_DATA_ROOT` (required): dataset root directory
- `OPENAI_EMBEDDING_API_KEY` (required): embedding key used by retrieval
- One LLM key depending on your selected `--llm`:
  - OpenAI: `OPENAI_API_KEY`
  - Google Gemini: `GOOGLE_API_KEY`
  - Anthropic Claude: `ANTHROPIC_API_KEY`

Example `.env` (minimum):

```env
OPENAI_EMBEDDING_API_KEY=YOUR_OPENAI_KEY_HERE
MULTIMATCHER_DATA_ROOT=YOUR_DATASET_ROOT_PATH

# Fill only what you use:
OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
# GOOGLE_API_KEY=YOUR_GOOGLE_KEY_HERE
# ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY_HERE
```

## Datasets Layout

Set `MULTIMATCHER_DATA_ROOT` so that the following folders exist under it:

- `M2Bench_Ecommerce/`
- `M2Bench_Healthcare/`
- `Unibench/`
- `M2E_Unibench/`

Expected structure:

```text
<MULTIMATCHER_DATA_ROOT>/
  M2Bench_Ecommerce/
    table/
    document/
    graph/
    grouping_candidates.csv
    group.csv

  M2Bench_Healthcare/
    table/
    document/
    graph/
    grouping_candidates.csv
    group.csv

  Unibench/
    table/
    document/
    graph/
    grouping_candidates.csv
    group.csv

  M2E_Unibench/
    table1/
    document1/
    graph1/
    table2/
    document2/
    graph2/
    grouping_candidates.csv
    group.csv
```

## Run

Basic usage:

```bash
python scripts/run_dataset.py --dataset <DATASET> --llm <LLM_ALIAS> --embedding-model text-embedding-3-large
```

### Supported datasets (`--dataset`)

- `m2bench-ecommerce`
- `m2bench-healthcare`
- `unibench`
- `m2e-unibench`

### Supported LLM aliases (`--llm`)

- `gpt-5`
- `gpt-5-mini`
- `gpt-oss-120b`
- `gpt-oss-20b`
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `claude-sonnet-4.5`
- `claude-haiku-4.5`
- `qwen3-max`
- `qwen3-next-80b`

### Examples

```bash
# M2Bench E-commerce + GPT-5
python scripts/run_dataset.py --dataset m2bench-ecommerce --llm gpt-5 --embedding-model text-embedding-3-large

# UniBench + Claude Sonnet 4.5
python scripts/run_dataset.py --dataset unibench --llm claude-sonnet-4.5 --embedding-model text-embedding-3-large

# Cross-dataset (M2Bench E-commerce <-> UniBench) + Qwen3-max
python scripts/run_dataset.py --dataset m2e-unibench --llm qwen3-max --embedding-model text-embedding-3-large
```

## Optional Flags

### 1) Override dataset root (`--data-root`)

If you don’t want to set `MULTIMATCHER_DATA_ROOT` in `.env`, pass it explicitly:

```bash
python scripts/run_dataset.py --data-root "C:\path\to\dataset" --dataset m2bench-ecommerce --llm gpt-5 --embedding-model text-embedding-3-large
```

If `--data-root` is a relative path, it is resolved relative to the repository root.

### 2) Override VectorDB path (`--vectordb-path`)

Default ChromaDB path:

- `<dataset gt_dir>/vectordb`

Override:

```bash
python scripts/run_dataset.py --dataset m2bench-ecommerce --llm gpt-5 --vectordb-path "C:\tmp\mm_vectordb"
```

### 3) Override LLM runtime params

```bash
python scripts/run_dataset.py --dataset m2bench-ecommerce --llm gpt-5 --temperature 1 --timeout 60 --max-retries 2
```

### 4) Override Kneedle D parameter (`--kneedle-d`)

S is fixed in code, and D can be overridden:

```bash
python scripts/run_dataset.py --dataset m2bench-ecommerce --llm gpt-5 --kneedle-d 0.85
```

## Output

The runner prints:

- dataset info and resolved paths
- number of schema contexts
- full raw LLM outputs (one per query element)

## Notes

- Keep `.env` out of Git (`.gitignore` should include `.env`).
- `vectordb/` is a generated artifact directory; ignore it if you don’t want to commit generated files.

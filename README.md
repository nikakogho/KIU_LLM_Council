# LLM Council

A small Python toolkit that runs a **multi-model “council”**:

1) models first **plan roles** (pick a default judge + solver set)  
2) solvers **draft** solutions  
3) solvers **peer-review** each other (strict JSON schema, with auto-repair)  
4) solvers **revise** their drafts using the peer feedback  
5) the judge **selects a winner** (strict JSON schema, with auto-repair + deterministic fallback)

Includes:
- a **CLI** (`llm_council_cli.py`)
- a **Tkinter GUI** (`llm_council_gui.py`)
- JSON **run persistence** to `runs/` (save/load/replay)
- provider clients for OpenAI, xAI, Anthropic, Gemini

---

## Requirements

- Python 3.10+ recommended
- API keys for at least **2 providers** (the roster is built from whichever keys are present)

---

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
````

---

## Configure environment (.env)

This repo uses `python-dotenv` and loads `.env` automatically via `llm_council/settings.py`.

1. Copy the example file:

```bash
cp .env.example .env
```

2. Edit `.env` and fill keys (leave blank if you don’t want that provider enabled):

```ini
# Keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
XAI_API_KEY=

# Cheap-ish defaults
OPENAI_MODEL=gpt-5-nano
ANTHROPIC_MODEL=claude-haiku-4-5
GEMINI_MODEL=gemini-2.5-flash-lite
XAI_MODEL=grok-3-mini
```

**Important:** The roster is created from whichever `*_API_KEY` variables are set.
You need at least **2 configured providers** to run a council.

---

## Entry points

### CLI (`llm_council_cli.py`)

The CLI takes the **problem statement as a positional argument**.

Basic run:

```bash
python llm_council_cli.py "Design a robust cache invalidation plan for a CDN-backed API."
```

Save run JSON to a specific file:

```bash
python llm_council_cli.py "Design a robust cache invalidation plan for a CDN-backed API." \
  --out runs/my_run.json
```

Print the winner text at the end:

```bash
python llm_council_cli.py "Design a robust cache invalidation plan for a CDN-backed API." \
  --print-winner
```

Show role opinions + judge scoring breakdown (debug):

```bash
python llm_council_cli.py "Design a robust cache invalidation plan for a CDN-backed API." \
  --show-opinions
```

Show judge parse/attempts + whether fallback was used:

```bash
python llm_council_cli.py "Design a robust cache invalidation plan for a CDN-backed API." \
  --show-judge
```

Skip review and/or revise phases:

```bash
python llm_council_cli.py "Design a robust cache invalidation plan for a CDN-backed API." \
  --no-reviews --no-revise
```

Override the judge provider (must be one of: `openai`, `anthropic`, `gemini`, `xai` AND present in the roster):

```bash
python llm_council_cli.py "Design a robust cache invalidation plan for a CDN-backed API." \
  --judge anthropic
```

Disable the interactive prompt (CLI otherwise may ask you to override the judge after showing the default plan):

```bash
python llm_council_cli.py "Design a robust cache invalidation plan for a CDN-backed API." \
  --no-interactive
```

**Default output path:** if you don’t pass `--out`, the run is saved to:
`runs/council_run_<UTC timestamp>.json`

---

### GUI (`llm_council_gui.py`)

Launch:

```bash
python llm_council_gui.py
```

GUI flow:

1. paste a problem statement
2. click **Plan roles**
3. pick a judge from the dropdown (the GUI requires an explicit selection to enable **Run council**)
4. click **Run council**
5. optionally **Save last run JSON** or **Load run JSON**

Options:

* **Skip reviews**
* **Skip revise**

Notes:

* The GUI auto-saves a completed run to the default `runs/council_run_<ts>.json`
* Loading a run shows the winner text + judge decision summary and repopulates judge options, but keeps selection empty (to force an explicit pick before running again)

---

## How it works (core logic)

### 1) Roster construction

`llm_council/runner.py` builds the roster from your `.env`:

* `openai` uses `OpenAICompatibleResponsesClient` with base URL `https://api.openai.com/v1`
* `xai` uses `OpenAICompatibleResponsesClient` with base URL `https://api.x.ai/v1`
* `anthropic` uses `AnthropicClient` (`https://api.anthropic.com/v1/messages`)
* `gemini` uses `GeminiClient` (`generativelanguage.googleapis.com`)

### 2) Role planning

Both CLI and GUI call:

* `llm_council.app.plan_council(...)`

This:

* asks each model for a strict-JSON role opinion (`llm_council.roles`)
* computes a deterministic default judge using priors + self-signal + nominations (`llm_council.role_planner.plan_roles`)
* optionally applies a user judge override *after* the default plan is computed (`apply_user_override`)

### 3) Council phases

Both CLI and GUI call:

* `llm_council.app.run_council(...)`

Phases (in order):

* `generate_drafts`
* `generate_peer_reviews` (optional)
* `revise_solutions` (optional)
* `judge_solutions`

During runs, both CLI and GUI receive streaming updates through the same callback bundle:

* `llm_council.app.RunCallbacks`

### 4) Strict JSON + auto-repair

Peer reviews and judge decisions must return **only JSON** (no markdown, no extra text).
If parsing fails, the engine retries once with a “repair” prompt.

Schemas:

* peer review: `PeerReviewJSON`
* judge decision: `JudgeDecisionJSON`

### 5) Deterministic fallback winner

If judge JSON is invalid (or names a provider not in solvers), the engine falls back to:

* average of parsed peer-review `overall` scores per solver
* if no parsed reviews exist, winner defaults to the **first solver in plan order**

---

## Persistence (runs/*.json)

Runs are saved/loaded via `llm_council/persistence.py`.

* `save_state(state, path)` writes a JSON artifact including:

  * `problem_statement`
  * `plan` (judge + solvers + opinions + score_breakdown)
  * drafts / reviews / revisions / judge output
  * `winner_provider`, `winner_text`
* `load_state(path)` rehydrates a `CouncilState`

Default filename format (UTC):

* `runs/council_run_YYYYMMDD_HHMMSS.json`

---

## Project structure

```
.
├── .env.example
├── llm_council_cli.py              # CLI entry point
├── llm_council_gui.py              # Tkinter GUI entry point
├── llm_council/
│   ├── app.py                      # shared orchestration (plan + run) used by CLI/GUI
│   ├── council_engine.py           # phases + strict JSON parsing/repair + fallback winner
│   ├── persistence.py              # save/load run JSON + default_run_path()
│   ├── role_planner.py             # role opinions + deterministic judge selection + override
│   ├── roles.py                    # role opinion schema + JSON extraction helpers + prompts
│   ├── runner.py                   # build clients/roster from settings
│   ├── settings.py                 # loads .env and exposes keys/models
│   ├── types.py                    # LLMReply dataclass
│   └── clients/
│       ├── base.py
│       ├── openai_compat.py         # used for openai + xai providers
│       ├── anthropic.py
│       └── gemini.py
├── runs/                            # saved JSON runs
├── tests/                           # (if present) pytest tests
└── requirements.txt
```

---

## Testing
Basic unit tests mocking the API calls and ensuring rest of the logic holds up

```bash
pytest -q
```

---

## Accuracy checks / charts

This section is intentionally a placeholder for the charts I plan to add.

Suggested charts to drop in here (images):

* Council vs single-shot success rate (by problem set)
* Win rate by provider
* Judge parse success rate / fallback usage rate
* Ablation: drafts-only vs +reviews vs +revise

<!-- INSERT CHART IMAGES BELOW -->

<!-- Example:
![](docs/charts/accuracy_overview.png)
![](docs/charts/fallback_rate.png)
-->

---

## Notes / gotchas

* You need **at least two API keys** set, otherwise roster size < 2 and runs are blocked.
* The OpenAI/xAI client uses the Chat Completions endpoint (`/chat/completions`).
* In `OpenAICompatibleResponsesClient`, if the model name starts with `gpt-5`, the payload removes `max_tokens` and `temperature`.

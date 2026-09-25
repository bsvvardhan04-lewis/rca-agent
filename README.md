# RCA Agent — Automated Root Cause Analysis for Incidents

> When a bug is reported, an agentic workflow is triggered to automatically query system
> logs, recent pull requests, and monitoring alerts associated with the timeframe of the
> bug, giving the developer a summarised hypothesis of what went wrong before they
> begin investigating.

A developer opening a ticket normally starts from zero: *when did this start, what
shipped recently, what else is alarming?* Those first ten minutes are mechanical, and
they are the same ten minutes every time. This agent does them.

**Runs entirely on self-hosted open-weight models. No API key, no third-party call.**
Incident data, log contents and source-code metadata never leave the network — which is
the point when the systems under investigation belong to a client.

---

## What it does

Input — a vague, human-written ticket:

> *"Support is getting a wave of tickets since roughly 20 minutes ago. Customers click Pay
> and get 'Something went wrong, please try again'. It is not everyone — some payments go
> through. Checkout page itself loads fine. Nobody on the team knows of a planned change."*

Output — a ranked, evidence-cited hypothesis set, written back to the ticket:

```
PR-4821 put a synchronous risk-service call inside the payments DB transaction,
so the 10-connection pool saturates under ordinary traffic.

1. Risk-scoring call inside tx_authorize exhausts the pool   ####################  90%
   evidence: L1, L4, C1, A2        against: —
2. Ledger retry-budget increase amplified load               #####...............  25%
   evidence: C4                    against: A3, L9
```

---

## How it works

```
  ticket (free text)
        |
        v
  [1] TRIAGE          one structured call
        |             -> services, error signatures, search window
        v
  [2] INVESTIGATION   agentic loop  OR  guided sequence
        |             -> logs / changes / alerts
        v
  [3] SYNTHESIS       one structured call
        |             -> ranked hypotheses, each citing evidence IDs
        v
  RCAReport  ->  Markdown / JSON / terminal
```

**Why three phases and not one loop.** A single loop that both investigates *and* fills
in a report schema stops investigating the moment it has enough to fill the schema.
Separating them means the investigator's only job is to find things, and the
synthesiser's only job is to be honest about what was found.

**Why the evidence ledger.** Every fact a tool returns is assigned a short ID (`L4`,
`C1`, `A2`). Hypotheses cite those IDs, and any citation that doesn't resolve is stripped
before the report is written. A plausible-sounding paragraph with no retrievable evidence
behind it is the main failure mode of this kind of tool; the ledger is what prevents it.

### Two investigation strategies

| Strategy | Who drives | Good for | Cost |
|---|---|---|---|
| `agentic` | The model picks each next tool call | 32B–70B instruct models; can follow a thread nobody anticipated | up to 20 generations |
| `guided` | Code runs the proven query sequence; the model interprets the results | small models, slow hosts, reproducible demos | 1 generation |
| `auto` *(default)* | Starts agentic, drops to guided the moment the model ignores the tools | mixed fleets | varies |

`guided` is not a crippled fallback. It executes exactly the method the investigator
prompt describes — count, narrow, read, correlate deploys, check alert fire order — as
code. A 7B model cannot reliably drive a six-tool loop, and twenty sequential generations
on a shared GPU is minutes of latency. This keeps the *reasoning* with the model and
moves the *procedure* into code.

### The tools the agent drives

| Tool | Why it exists |
|---|---|
| `list_services` | Discover valid service names before filtering on them |
| `log_volume` | **Counting is cheap, reading is expensive.** Find *when* it started across thousands of lines without pulling them into context |
| `search_logs` | Pull a small representative sample from a narrow window; every line gets an evidence ID |
| `list_recent_changes` | PRs that reached production in the window — correlated on **deploy** time, not merge time |
| `get_change_detail` | File list + the author's own description, to test whether a mechanism actually explains the errors |
| `list_alerts` | Alert **fire order** is a causal signal: whoever alerted first is usually nearer the cause |

---

## Model setup

Everything self-hosted speaks the OpenAI chat-completions format, so one client covers
**vLLM, Ollama, TGI, LM Studio and llama.cpp**. Changing model or host is a config change.

### Production — vLLM on the GPU box

```bash
vllm serve Qwen/Qwen3-32B --port 8000
```

```bash
RCA_MODEL=Qwen/Qwen3-32B
RCA_BASE_URL=http://gpu-host:8000/v1
RCA_STRATEGY=agentic
```

`Qwen3-32B` and `Llama-3.3-70B-Instruct` both handle multi-step tool calling well. That
matters here: the design leans on it.

### Local development — Ollama

```bash
ollama serve
ollama pull qwen2.5:7b
```

```bash
RCA_MODEL=qwen2.5:7b
RCA_BASE_URL=http://localhost:11434/v1
RCA_STRATEGY=guided
```

On a CPU-only laptop, use `guided` — one generation instead of twenty.

> **On small models.** Llama 3.2 (1B/3B) technically supports tool calling, but a
> six-tool loop with a long system prompt is past what it does reliably. It works in
> `guided` mode. For `agentic`, use 7B minimum and preferably 32B+.

Structured output uses guided decoding (`response_format: json_schema`) where the server
supports it, and degrades automatically where it doesn't. Every structured reply is
validated against its Pydantic schema with one repair round, because open-weight models
drop required fields more often than hosted ones.

---

## Quick start

```bash
cd D:\Projects\rca-agent
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

```bash
python scripts/seed_demo.py
python -m rca.cli sources
python -m rca.cli investigate INC-1042 --verbose
```

`sources` costs nothing and tells you whether the model server is reachable, which
services have logs, and which strategy is active.

Ad-hoc, without a ticket file:

```bash
python -m rca.cli investigate --title "Payments 500s" --description "Customers can't pay since ~20m ago"
```

The Markdown report lands in `reports/INC-1042.md`. Add `--json-out path.json` for the
full structured report.

---

## Triggering it automatically

```bash
uvicorn api.server:app --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /incidents` | File an incident, get a job ID back immediately |
| `GET /jobs/{id}` | Poll status, progress, and the finished report |
| `GET /jobs/{id}/markdown` | The report, ready to paste onto the ticket |
| `POST /webhooks/jira` | Accepts a Jira `issue_created` payload (incl. ADF descriptions) |
| `POST /webhooks/sentry` | Accepts a Sentry issue-alert payload |
| `GET /health` | Sources, model server reachability, active strategy |

An investigation takes tens of seconds to minutes — longer than any webhook sender will
wait — so `POST` returns `202` with a job ID rather than blocking.

`/health` reports `degraded`, not `ok`, when the model server is unreachable: the sources
still work, but no investigation can actually run, and a readiness probe should say so.

> The job store is in-process and non-durable. Fine for one worker and a demo; back it
> with Redis or a table and move the agent into a worker process for real use.

---

## Scenarios and evaluation

`scripts/scenarios.py` defines four incidents, each with recorded ground truth.
They are chosen to punish one specific shortcut:

| Scenario | Root cause | What it tests |
|---|---|---|
| `INC-1042` | A recent deploy (`PR-4821`) | The obvious case — correct deploy correlation |
| `INC-1043` | External provider 503s | **No deploy is responsible.** A decoy PR shipped 95 min earlier |
| `INC-1044` | Disk filled over days | No deploy, no sharp onset — a gradual trend |
| `INC-1045` | A feature flag flipped | Code shipped 3 days ago and sat inert until the flag |

**Why three of the four have no culpable deploy.** An agent that always blames the most
recent PR scores 100% on `INC-1042` and is actively harmful in production — it sends an
engineer to revert innocent code while the real fault continues. Three-quarters of the
set exists to catch exactly that.

```bash
python scripts/seed_demo.py --all
python scripts/evaluate.py
python scripts/evaluate.py --strategy guided --model qwen2.5:7b
```

Each report is scored on eight checks. Two are critical — `cause` and `decoys` — and a
run only passes if both hold: getting the headline right while failing to dismiss the
decoy is not a pass. Results are written to `reports/eval-<timestamp>.json` so two runs
can be compared after a prompt change. Without that, "the prompt feels better now" is the
only evidence available, which is not evidence.

The scenarios are themselves tested ([`tests/test_scenarios.py`](tests/test_scenarios.py)):
a scenario that quietly stops being hard would make the eval report green while measuring
nothing. One of those tests already caught a decoy deploy sitting 17 minutes before onset,
which would have made the ground truth unfair rather than the case hard.

### The demo scenario in detail

`INC-1042` generates ~2,400 log lines across four services. It is built to be *hard*
rather than illustrative:

- The real cause is a PR whose description sounds entirely reasonable.
- A decoy deploy landed 110 minutes earlier, on the service the ticket actually names.
- A critical-looking alert has been firing since long before the incident.
- A chronic slow-query warning runs through the whole window.
- The ticket's `service_hint` points at the wrong service — where the symptom was *seen*,
  not where the fault is.

Re-run the seeder when the data goes stale — it anchors timestamps to the current time.

---

## Swapping in real data sources

Everything file-backed sits behind a protocol in [`rca/sources/base.py`](rca/sources/base.py).
To point this at production, write one class — the agent doesn't change:

| Protocol | Ships with | Implement against |
|---|---|---|
| `LogSource` | JSONL files | Loki, Elasticsearch, CloudWatch, Datadog Logs |
| `ChangeSource` | JSON files, **real git** | GitHub/GitLab PR API |
| `AlertSource` | JSON files | Prometheus Alertmanager, Datadog Monitors, PagerDuty |

### Pointing it at a real repository

Set `RCA_GIT_REPO` to a local clone and [`GitChangeSource`](rca/sources/git_source.py)
replaces the demo change file. It walks `git log` for the window, pulls the file list and
line counts from each diff, recovers PR numbers from squash-merge subjects
(`Add risk scoring (#4821)` → `PR-4821`), and infers services from monorepo paths.

> **Git does not know about deploys.** It knows when a commit landed on a branch, not when
> it reached production — and the correlation depends on deploy time. So `deployed_at` is
> `None` and it falls back to merge time, meaning code merged Friday and deployed Monday
> looks like a Friday change. Pass a real deploy feed as `deploy_times` and the
> correlation is accurate again.

A misconfigured `RCA_GIT_REPO` degrades to the file source rather than refusing to start.
Pass `strict=True` to [`build_sources`](rca/sources/__init__.py) if you'd rather it fail loudly.

### One rule worth knowing

When the agent filters changes by service, changes whose service **can't be determined**
(a root config, `CODEOWNERS`, a shared library) are *not* hidden — a shared file can break
any service, and silently dropping it mid-incident is the worse failure. They come back
tagged `UNATTRIBUTED` so the agent weighs them instead of reading them as confirmed hits.

---

## Layout

```
rca/
  agent.py          three-phase orchestration, both strategies
  prompts.py        the system prompts, isolated for tuning
  tools.py          the six tools
  evidence.py       the ledger that makes citations resolvable
  models.py         domain models + the schemas the model fills
  report.py         Markdown and terminal rendering
  evaluation.py     scoring a report against ground truth
  cli.py            typer CLI
  llm/
    base.py         LLMClient protocol, ToolSpec - no vendor imports
    openai_compat.py  vLLM / Ollama / TGI / LM Studio
    schema.py       JSON schema from signature + docstring
    structured.py   schema validation with one repair round
  sources/
    base.py         LogSource / ChangeSource / AlertSource protocols
    file_sources.py offline implementations
    git_source.py   real git repositories
api/server.py       FastAPI webhook trigger
scripts/
  scenarios.py      four incidents + their ground truth
  seed_demo.py      writes scenarios to disk
  evaluate.py       runs the sweep and scores it
tests/              138 tests, no server needed
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `RCA_MODEL` | `qwen2.5:7b` | Production: `Qwen/Qwen3-32B` or `Llama-3.3-70B-Instruct` |
| `RCA_BASE_URL` | `http://localhost:11434/v1` | Ollama; vLLM is usually `:8000/v1` |
| `RCA_API_KEY` | `not-needed` | Self-hosted servers ignore it |
| `RCA_STRATEGY` | `auto` | `agentic` / `guided` / `auto` |
| `RCA_TIMEOUT` | `600` | Generation timeout, seconds |
| `RCA_JSON_SCHEMA` | `true` | Guided decoding; set false for older servers |
| `RCA_MAX_TOOL_ITERATIONS` | `20` | Ceiling on the agentic loop |
| `RCA_MAX_LOG_LINES` | `60` | Hard cap on lines any one search returns |
| `RCA_DATA_DIR` | `./data` | |
| `RCA_GIT_REPO` | — | A local clone; swaps the demo change file for real git |
| `RCA_GITHUB_REPO` | — | `owner/repo`, used only to build PR links |

## Tests

```bash
pytest
```

No server and no credentials required. The agent tests script the client but run the
**real** tools against the **real** seeded data, so everything except the model's
judgement is covered: schema generation, tool dispatch, argument handling, evidence IDs,
citation resolution, strategy selection, the webhook adapters, and rendering.

---

## Limits worth knowing

- It reads change *metadata* — titles, descriptions, file lists — not diffs. It can say
  "this PR touches the transaction path", not "line 40 is wrong".
- Confidence is the model's own calibration. Treat it as a ranking signal, not a
  probability.
- It is a starting point for a human, not a verdict. The report leads with
  `verification_steps` for exactly that reason.

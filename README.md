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
ollama pull llama3.2:3b
```

```bash
RCA_MODEL=llama3.2:3b
RCA_BASE_URL=http://localhost:11434/v1
RCA_STRATEGY=auto
```

**Which strategy on a CPU-only laptop?** Measured on `llama3.2:3b` (i5-12500H, no GPU) against
INC-1042, one run each:

| Strategy | Time | Outcome |
|---|---|---|
| `agentic` | 541 s | Named PR-4821, the true cause, but with a vague mechanism |
| `guided` | 645 s* | Gathered more evidence and wrote a better timeline, but blamed the decoy PR-4817 and never named PR-4821 |

\* Run alongside other work, so not a clean timing. Even so, `guided` is not the speed-up it
looks like: most of the time is the final report-writing generation, not the tool loop.
A follow-up chat answer takes roughly 10–20 s once the model is loaded; the first one after
loading can take a minute or more. One incident, one run each - not a benchmark.

**The timeline is built from the evidence, not written by the model.** The 3B model's own
timeline put a log line's time on an alert, listed a successful `200` as an incident event
and repeated an entry. `rca/timeline.py` builds it from the ledger instead - every change and
alert, repeated log lines collapsed with a count, routine INFO traffic dropped once real
errors exist - and each line ends with the ID of the evidence behind it. It says what
happened when; it does not explain why. (The evaluator ignores the timeline when checking
whether a decoy was "considered", since a mechanical timeline lists every change retrieved.)

**Stated confidence is guarded in code.** Both runs came back at 90%, one of them wrong. A
hypothesis rated above 0.85 is lowered to 0.70 - and the reason written into its reasoning -
when it is the only hypothesis considered, cites no support, or has evidence cited against
it (`rca/calibration.py`). It cannot judge whether a mechanism is well explained, so a
confident answer with a vague mechanism still gets through.

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
| `GET /health` | Sources, model server reachability, active strategy. `?deep=true` also runs a 4-token generation |
| `POST /chat` | Ask about the standing report: `{"message": "..."}` in, `{"reply": ...}` out. Add `"history": [...]` and it is fully stateless |
| `POST /v1/chat/completions` | The same, in the OpenAI chat format (non-streaming, stateless) |
| `GET /v1/models` | One model, `rca-agent`; OpenAI-format clients call this as a connectivity check |
| `GET/POST/DELETE /jobs/{id}/chat` | The same, scoped to one finished job, with its transcript |
| `GET /chat` | The chat console (one HTML page, no build step, no CDN) |

An investigation takes tens of seconds to minutes — longer than any webhook sender will
wait — so `POST` returns `202` with a job ID rather than blocking.

`/health` reports `degraded`, not `ok`, when the model server is unreachable: the sources
still work, but no investigation can actually run, and a readiness probe should say so.

> The job store is in-process and non-durable. Fine for one worker and a demo; back it
> with Redis or a table and move the agent into a worker process for real use.

---

## Follow-up chat

A report is a push; the first thing a developer does with it is argue back — *why that
PR and not the other? how sure are you? what didn't you check?* The chat answers from the
evidence the investigation already collected. It has no tools and no source access, so it
cannot contradict the report on screen.

```bash
# 1. Save a finished report (this is what the chat will be about)
python -m rca.cli investigate INC-1042 --json-out data/demo/report.json

# 2. Serve it. RCA_SEED_DEMO=1 opens the console on that report immediately.
RCA_SEED_DEMO=1 uvicorn api.server:app --port 8000
#    console:  http://localhost:8000/chat

# 3. Or call it as an API
curl -X POST localhost:8000/chat -H "Content-Type: application/json"      -d '{"message": "Why that PR?"}'
#    pass the returned session_id back to continue the thread
```

`POST /chat` answers about `RCA_CHAT_REPORT` (default `data/demo/report.json`) so a caller
needs no job ID. A reply carries three checks, because a chatbot's citations are only
useful if they can't be faked:

| Field | What it guards against |
|---|---|
| `unverified` | The model cited an evidence ID that does not exist (`[I1]`, `[L99]`). It is rewritten to `[unverified]` in the reply rather than left looking real. |
| `unsupported_times` | A timestamp in the answer that the *cited* evidence does not contain. Added after a live 3B model attributed a pull-request deploy time to an alert. The reply gets a visible `[Check: …]` note. |
| `unsupported_counts` | A count the ledger cannot support: "two alerts" when one was retrieved, or "another change" when only one was. Only over-counting is flagged, and negated phrasing ("no other change") is not. Added after a live 3B model answered "two" and invented the second from a PR's deploy time. |
| scope refusal | Anything not about the incident gets one fixed sentence. Verified on `llama3.2:3b` and `llama3.1:8b`. |

### For an evaluation harness

A generic evaluator has one URL and a message; it cannot file an incident and wait for a
job. Two stateless shapes cover it — use whichever its config speaks:

```bash
# plain JSON
curl -X POST localhost:8000/chat -H "Content-Type: application/json"      -d '{"message": "Why that PR?", "history": []}'

# OpenAI format - works with the openai SDK / any OpenAI-compatible tool
curl -X POST localhost:8000/v1/chat/completions -H "Content-Type: application/json"      -d '{"model": "rca-agent", "messages": [{"role": "user", "content": "Why that PR?"}]}'
```

The caller owns the conversation: send the turns you want considered each time and the
server keeps nothing. Any `system` message in the request is dropped on purpose — the
grounding and scope rules belong to the service, not to the caller. `stream: true` is
refused with a clear error; `temperature` and `max_tokens` are ignored. The OpenAI reply
adds an `rca` object (`citations`, `unverified`, `unsupported_times`) that clients which
do not know it will ignore.

**Serving it to anyone but yourself.** Do not tunnel the main app: it also lists every
job and report and accepts incident filing. Run the public app, which carries only the
chat routes, and give it a token:

```bash
RCA_CHAT_TOKEN=<long random string> uvicorn api.server:public_app --port 8001
# callers send:  Authorization: Bearer <token>   (or X-API-Key: <token>)
```

`public_app` fails closed: with no `RCA_CHAT_TOKEN` it answers 503 to everything
(`RCA_PUBLIC_OPEN=1` opts out explicitly). The same token also guards these routes on the
main app when set; unset, the main app stays open for localhost use.

**Cold start.** On a CPU the first answer after the model loads took 87-106 s in testing and
later ones 10-20 s; a harness with a request timeout would see the first call fail. Start the
server with `RCA_WARMUP=1` and it answers one real question about the report at startup, then
repeats every `RCA_WARMUP_INTERVAL` seconds (default 240, `0` = once) so the model is not
unloaded for idleness. Off by default - it spends CPU.

**Known limits, stated plainly.** The checks prove a cited ID *exists* and that stated
times are *present in what was cited*; they cannot prove a sentence is true, and they do
not see times written as "around 8am". The chat can also only be as good as the report it
reads — a weak report gives a weak chat.

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


### Measured results (llama3.2:3b, CPU only)

Run with `python scripts/evaluate.py` and `python scripts/baseline_single_call.py`; the JSON for
each run is written to `reports/` (generated output, so not committed). Four scenarios, scored by the same eight checks.

| Configuration | Passed | Mean score | Citations valid | Time / run |
|---|---|---|---|---|
| Single model call, no tools (baseline) | 0 / 4 | 48% | 0 / 4 | 107 s |
| Pipeline, before the coverage guard | 0 / 4 | 42% | 1 / 4 | 268 s |
| Pipeline, with the coverage guard | 1 / 4 | 46% | 4 / 4 | 198 s |

A repeat run with the guard gave identical results. A first run of that sweep was discarded
for timing (the laptop slept mid-run and recorded durations of 1,538 s and 8,144 s); its
quality results matched the clean run exactly.

What this does and does not show. The first full sweep found that three of four runs made a
single tool call, because the loop ends when the model stops asking, so no change list was ever
read. The coverage guard fixes that: code now fetches any evidence source the model skipped
(`RCAAgent._ensure_coverage`). It improved evidence use (4 to 7 tool calls, every citation
valid) but **not accuracy**: on this 3B model the pipeline is still level with a single call
(46% vs 48%), and with the change list in hand it blamed a decoy PR on one scenario, which is
the newest-PR trap the scenarios exist to catch. "Cause right" is high for the baseline only
because three of four scenarios have no culprit, so blaming nothing is correct. The design
assumes a 32B to 70B model; that has not been run. Four scenarios, written by us, is a small
and non-independent test.

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
api/server.py       FastAPI: webhooks, jobs, follow-up chat, evaluator routes
  api/static/chat.html  the chat console
rca/chat.py         grounded Q&A over a finished report
rca/timeline.py     timeline built from the evidence ledger
rca/calibration.py  caps unearned high confidence
scripts/
  scenarios.py      four incidents + their ground truth
  seed_demo.py      writes scenarios to disk
  evaluate.py       runs the sweep and scores it
tests/              264 tests, no server needed
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `RCA_MODEL` | `llama3.2:3b` | Production: `Qwen/Qwen3-32B` or `Llama-3.3-70B-Instruct` |
| `RCA_BASE_URL` | `http://localhost:11434/v1` | Ollama; vLLM is usually `:8000/v1` |
| `RCA_API_KEY` | `not-needed` | Self-hosted servers ignore it |
| `RCA_STRATEGY` | `auto` | `agentic` / `guided` / `auto` |
| `RCA_TIMEOUT` | `600` | Generation timeout, seconds |
| `RCA_JSON_SCHEMA` | `true` | Guided decoding; set false for older servers |
| `RCA_CHAT_REPORT` | `data/demo/report.json` | The report `POST /chat` answers about |
| `RCA_SEED_DEMO` | off | `1` lists that report as a finished job so the console opens on it |
| `RCA_CHAT_TOKEN` | unset | Bearer token for `POST /chat` and `/v1/*`. Required by `public_app` |
| `RCA_WARMUP` | off | `1` answers one real question at startup so the first caller is not the one who waits |
| `RCA_WARMUP_INTERVAL` | `240` | Seconds between keep-warm questions; `0` warms once |
| `RCA_PUBLIC_OPEN` | off | `1` lets `public_app` serve with no token (not recommended) |
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

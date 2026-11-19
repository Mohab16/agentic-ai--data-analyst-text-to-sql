# AI Data Analyst (Agentic Text-to-SQL)

An LLM investigates a business database and answers questions with evidence, not
templates. Every number in an answer traces back to a SQL statement the model
wrote, which a code-enforced guard validated, which a real database engine
executed.

**The central principle:** the LLM is the analyst, the database is its evidence,
the SQL tools are its interface. There are no `get_monthly_revenue()` helpers.

---

## Status: Phase 6 of 8

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Contracts, configuration, context builder, trace, orchestrator skeleton, CLI | done |
| 1 | Mock SQLite warehouse (seed data, schema digest, `inspect_schema`) | done |
| 2 | SQL validation boundary + `execute_sql` / `get_query_error_details` | done |
| 3 | LLM client + planner agent | done |
| 4 | Investigator (bounded tool loop, evidence-linked findings) | done |
| 5 | Reporter + full CLI output | done |
| 6 | Structured-JSON fallback for models without tool calling | done |
| 7 | PostgreSQL adapter (disabled by default) | done |
| 8 | Golden evaluations, README completion, dependency audit | pending |

**What works now:** configuration validation, plan construction and validation,
context assembly with a hard budget, append-only JSONL traces with secret
redaction, the step loop with wall-clock enforcement, a seeded read-only SQLite
warehouse, schema introspection, the schema digest + `inspect_schema` tool, a
code-enforced SQL guard, real SQL execution with a row cap and a statement
timeout, failure forensics by `error_id`, an OpenAI-compatible client with typed
failures and bounded retries, a planner whose JSON is validated in code, and an
investigator that writes real SQL through a bounded tool loop and returns claims
that each cite the query that supports them.

**What does not exist yet:** the golden evaluations (phase 8). A question runs all
three real steps and prints the plan, the investigation, the prose answer with
assumptions and confidence, and the run facts.
---

## The mock dataset

Built deterministically from `MOCK_SEED`, `MOCK_MONTHS`, and `REFERENCE_DATE`.
Six related tables (`customers`, `products`, `orders`, `order_items`,
`payments`, `returns`), ~6,000 orders over 24 months, ending yesterday.

The data contains a story worth investigating, planted by construction:

- revenue falls in the most recent *complete* month, driven by an Electronics
  shortfall;
- Electronics revenue in the **North** region is exactly zero for that month;
- Fashion return rates spike in the same month;
- the current month is **partial**, so comparing it with a full month shows a
  fake collapse. The schema digest says so explicitly.

Because the shortfall is applied positionally (every Nth item) rather than
probabilistically, the story holds for every seed — a fixture that only works on
one seed is a fixture that breaks silently.

```powershell
python -m app.main --show-schema            # what the planner will receive
python -m app.main --inspect-table orders   # what the model can ask for
```

---

## SQL safety

Two independent layers, both in code. A prompt cannot grant either of them.

**1. The guard (`db/guard.py`)** parses the statement with sqlglot and fails
closed. Refused: multiple statements, anything that is not a query (writes, DDL,
`PRAGMA`, `ATTACH`, transactions, `COPY`, `MERGE`, `GRANT`, `SET`, `EXPLAIN`),
`SELECT ... INTO`, row-locking reads, data-modifying `WITH` clauses, oversized
input, and functions that touch files, the network, or the server
(`load_extension`, `readfile`, `pg_read_file`, `dblink`, `set_config`, `nextval`,
...).

The `WITH` case is why the guard walks the syntax tree instead of trusting the
statement type: sqlglot parses
`WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x` as a plain `Select`.

**2. The engine (`db/mock_source.py`)** denies writes, DDL, `PRAGMA`, `ATTACH`,
and explicit transactions through a SQLite authorizer installed after seeding. A
statement that somehow slipped past the guard still cannot change data — there is
a test that calls the executor directly with `DELETE FROM orders` and asserts the
engine refuses it.

Two more properties worth stating plainly:

- **The guard never rewrites SQL.** The statement that executes is the statement
  the model wrote, minus a trailing semicolon. Rendering the parsed tree back to
  text is lossy — sqlglot's SQLite generator drops CTE column lists, which turned
  a valid recursive query into an invalid one during development. The row cap is
  enforced when fetching (`MAX_ROWS + 1`, so truncation is detectable), and the
  statement timeout is a progress handler.
- **Errors are values, not exceptions.** `execute_sql` returns a short message
  plus an `error_id`; `get_query_error_details(error_id)` returns the driver
  message, the failing SQL, and the columns of every table it referenced, so a
  model can repair a query instead of guessing. A query that returns zero rows is
  a success, and says so.


---

## How a model call works

`--plan` runs only the first step, which makes it the cheapest way to see the model
boundary. It is the real first step of the workflow, not a demo - the same `Step`,
the same context builder, the same orchestrator, the same trace, and the same
artifact the investigator reads in the next step.

1. **Context is assembled, not accumulated.** The `plan` step declares two reads
   (`question`, `schema`), and `core/context.py` concatenates exactly those,
   truncated to `CONTEXT_BUDGET` if needed. The static instruction lives in the
   agent, never in the assembled text (X-06).
2. **The client sends one request.** Model, messages, `max_tokens`, optional
   `response_format={"type": "json_object"}`. Temperature is not sent: some
   providers reject anything but their default, and the planner's determinism
   comes from validation, not sampling.
3. **The reply is validated in code.** The JSON is parsed against the plan
   schema: dates must be real, `start <= end`, granularity is one of three values,
   at most `MAX_QUERIES` checks, ids are unique slugs, an unanswerable plan must
   name what is missing and list no checks. Every problem is collected, not just
   the first.
4. **One repair turn.** A rejected payload goes back to the model with the list of
   problems. A second failure ends the step with `plan_invalid` — a
   plausible-looking plan that silently violates the contract is worse than none.
5. **Failures are kinds, not strings.** `auth`, `rate_limit`, `timeout`,
   `connection`, `bad_request`, `not_found`, `server`, `empty_response` — each with
   a `retryable` flag. The client owns retries (`LLM_MAX_RETRIES`, default 1) and
   disables the SDK's, so attempts are countable and traceable.

What the planner is *not* allowed to decide: the workflow, the schema, the SQL, or
whether the answer is supported. It proposes an investigation; later steps execute
it and check the evidence.

---

## How the investigation works

A question runs two steps of one task: `plan`, then `investigate`. The second step
is a bounded tool loop over three tools and nothing else.

| Tool | What the model gets | Costs |
| --- | --- | --- |
| `execute_sql` | the guarded statement's rows, its `query_id`, typed failures | one query |
| `inspect_schema` | one table's columns, value domains, sample rows | nothing |
| `get_query_error_details` | driver message, failing SQL, columns of referenced tables | nothing |

The loop ends in one of three ways and the report says which (`stop_reason`): the
model stops calling tools (`model_finished`), it spends `MAX_TOOL_TURNS` turns
(`turn_budget`), or it exhausts a query budget (`query_budget`). The last two still
get one final turn *without* tools, so a cut-short investigation reports what it
has instead of nothing.

Budgets are enforced where the counter is, in `tools/registry.py`, not in a prompt:

- `MAX_QUERIES` (8) bounds statements that reached the database. A guard refusal
  never reaches it, so it is charged to the failure allowance instead.
- `MAX_QUERY_RETRIES` (2) bounds failures: one attempt plus that many retries.
  Without it a model can spend the whole query budget resending one broken
  statement, and the trace shows eight failures instead of a finding.

**Findings are checked, not trusted.** The model ends by returning one JSON object
of claims, each with `evidence_query_ids`. The parser accepts an id only if a query
with that id actually returned rows: a failed query, a refused query, and a query
that never ran are all rejected, and the rejection goes back once with every
problem listed. "No claim without evidence" is therefore a property of the code
path, not a sentence in a prompt.

What that check does *not* do: it does not re-add the numbers. A claim can cite a
real query and misquote its rows. Arithmetic verification is what the golden
evaluations in phase 8 are for; in the meantime every `QueryRecord` carries the
`query_id`, the SQL that ran, and a bounded preview of the rows, so a verifier has
something to check against.

Every executed statement also becomes a `QueryRecord` and stays in the result,
including failures and budget refusals. That record set is what the reporter will
cite, what `--json` exposes, and what the CLI prints as the SQL trail. The JSONL
trace keeps one event per step (with the tool names called and the token counts);
the SQL itself lives in the artifact, because agents do not write traces by design.

---

## HTTP API

A FastAPI server serves the same pipeline over HTTP (port 8001, Swagger UI at
`/docs`):

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{"status": "ok", "version": "0.1.0"}` |
| `POST` | `/answer` | `{"question": "..."}` | plan + investigation + prose answer |
| `POST` | `/plan` | `{"question": "..."}` | validated `InvestigationPlan` only |
| `POST` | `/investigate` | `{"question": "..."}` | `InvestigationReport` (queries + findings) |

```powershell
python -m app.main --serve --host 127.0.0.1 --port 8001
Invoke-WebRequest -Uri http://127.0.0.1:8001/answer `
  -Method Post -ContentType application/json `
  -Body '{"question":"Why did revenue drop last month?"}'
```

Blocking LLM calls run via `asyncio.to_thread` so the event loop stays responsive.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env    # then fill LLM_API_KEY / LLM_MODEL
```

Requires Python 3.12+.

---

## Commands

```powershell
# Validate configuration and print the plan. No model call, no credentials needed.
python -m app.main --dry-run "Why did our revenue drop this month?"

# Same, machine-readable.
python -m app.main --dry-run --json "Why did our revenue drop this month?"

# One real model call: question + schema in, validated plan out.
python -m app.main --plan "Why did our revenue drop this month?"
python -m app.main --plan --json "Why did our revenue drop this month?"

# The real thing: plan, then investigate the mock warehouse with SQL.
python -m app.main "Why did our revenue drop this month?"
python -m app.main --json "Why did our revenue drop this month?"

# Inspect the mock warehouse.
python -m app.main --show-schema
python -m app.main --inspect-table orders

# Run one statement exactly as the agent will: guarded, capped, timed.
python -m app.main --query "SELECT COUNT(*) AS n FROM orders"
# On failure, also print get_query_error_details output.
python -m app.main --query "SELECT revenu FROM orders" --explain

# Override the trace directory for one run.
python -m app.main --dry-run --trace-dir .\traces\demo "Question here"

# Tests.
python -m pytest
```

Exit codes: `0` ok, `2` usage, `3` configuration, `4` plan invalid,
`5` backend requested but not available in this build, `6` SQL refused by the
guard, `7` SQL failed in the database, `8` an agent or model call failed. On `8`
the CLI still prints the SQL that ran before the failure.

---

## Layout

```text
app/main.py             CLI: argv, .env loading, Settings, dispatch, exit codes
app/render.py           terminal rendering (plain text, no markup dialects)
wire.py                 composition root: database, model client, agent map
core/types.py           all contracts + JSON serialization
core/config.py          Settings: one validated object per process
core/context.py         build_context(step, artifacts, budget) with truncation
core/orchestrator.py    the step loop: validate, dispatch, budget, trace, state
core/trace.py           append-only JSONL, one file per task, redacted
db/base.py              adapter shapes + the Database protocol + error journal
db/seed.py              deterministic dataset generation and DDL
db/mock_source.py       SQLite build, introspection, engine-level read-only lock
db/guard.py             SQL validation: proof that one statement is read-only
db/factory.py           backend selection
tools/schema.py         schema digest + inspect_schema tool
tools/sql.py            execute_sql + get_query_error_details + rendering
tools/registry.py       tool dispatch: specs, arguments, budgets, evidence records
agents/replies.py       reading a JSON reply + the wording of the one repair turn
agents/planner.py       question -> validated InvestigationPlan, one repair turn
agents/investigator.py  bounded tool loop -> InvestigationReport, claims cite queries
llm/base.py             request/response shapes, typed failures, the client protocol
llm/openai_client.py    the only provider-specific file: chat completions + retries
llm/structured_client.py  adapts tool requests for models without tool calling
tests/                  unit tests, bypass corpus, story assertions, import guard
```

Not here yet, by design: `evals/`, the PostgreSQL adapter.

Planned workflow (three steps, fixed order, one agent each):

```text
plan         planner       question + schema        -> InvestigationPlan
investigate  investigator  + plan                   -> InvestigationReport (queries + findings)
answer       reporter      + report                 -> FinalAnswer (prose + evidence ids)
```

Autonomy lives only inside `investigate`, as a bounded tool loop. Everything
else is an ordered, testable pipeline.

---

## Design rules in force

- An agent is a plain function. No base class, no registry, no framework.
- Routing is a dict lookup on `Step.agent`. The agent is never parsed from prose.
- Failures are typed values: `ToolError` for recoverable tool failures,
  `AgentResult(status="error")` for agent failures. Error text is never a
  deliverable.
- Context is built from declared reads only and truncated deterministically; the
  trace records the truncation.
- Configuration is read once, in one place, and validated with every problem
  reported in a single message.
- SQL validation will be code, not a prompt instruction, with the database
  engine's own read-only mode underneath it.
- No business logic in tools. No predefined analysis methods. Ever.
- One module knows concrete implementations (`wire.py`). Everything else
  depends on contracts or protocols, and a test enforces the import graph.

See `DECISIONS.md` for the decisions behind these and the escapes from the
default methodology.

---

## Trace format

One JSON object per line, one file per task, at `traces/<task_id>.jsonl`:

```json
{"agent": "planner", "duration_ms": 4120, "error": null, "model": "...", "outcome": "ok", "step_id": "plan", "task_id": "tsk_...", "tokens_in": 0, "tokens_out": 0, "tool_calls": []}
```

Secrets (API key, DSN password) are redacted before writing.

# Decisions

Deliberate choices that shape the code, including deviations from the
methodology the project is built on (`Agentic AI Engineering` skill). Undocumented
deviation is how anti-patterns creep back in, so every entry here is a decision
someone can revisit, not an oversight.

---

## Locked decisions

### D-01 — Three agents, not six stages

The specification names six stages (understand, inspect, plan, generate,
evaluate, answer). Implemented as three roles, because a role is defined by its
input/output contract, not by its name:

| Spec stages | Role | Contract |
| --- | --- | --- |
| 1 + 3 | `planner` | question + schema -> `InvestigationPlan` |
| 4 + 5 + 6 + 7 | `investigator` | plan + schema -> `InvestigationReport` (bounded tool loop) |
| 8 | `reporter` | report -> `FinalAnswer` |

Stage 2 (schema inspection) is a tool plus a deterministic pre-fetch, not an
agent. Splitting further would create agents with identical shapes. Rule: A-02,
A-03.

### D-02 — Schema digest is pre-fetched, `inspect_schema` stays available

The orchestrator fetches a bounded schema digest before step 1 and stores it as
the root artifact `schema`. It costs nothing and every downstream step needs it.
`inspect_schema(table)` remains a tool inside the investigation loop for DDL,
sample rows, and low-cardinality distinct values, which is what makes generated
SQL correct in practice. Rule: X-01, X-03.

### D-03 — `AgentResult.data` carries structured payloads

The methodology's minimal contract has `text` only. Planning and investigation
outputs are structured and consumed by code, so stuffing them into `text` as
JSON would trade type safety for nothing. `data: dict | None` is additive with a
default, JSON-serializable, and round-trip tested. Rule: C-01, C-05.

### D-04 — `db/` is a capability adapter that `tools/` may import

The starter template forbids `tools/*` from importing "the database layer",
meaning persisted *task state*. Here the database is the external system the
tools exist to access, exactly like an HTTP client. The adapters live in `db/`
and `tools/sql.py` imports them. Task state remains owned by the orchestrator
alone. Rule: L-01, T-05 (spirit), starter template section 3 (explicit
deviation).

### D-05 — SQL safety is two independent layers

1. `db/guard.py` parses each statement with `sqlglot` in the target dialect and
   rejects anything that is not a single read-only `SELECT`/`WITH ... SELECT`,
   including data-modifying CTEs, `SELECT INTO`, `ATTACH`, `PRAGMA`, `COPY`,
   `CALL`, `DO`, and known side-effectful functions. Parse failure means reject.
2. The engine enforces read-only independently: SQLite via
   `Connection.set_authorizer`, PostgreSQL via a read-only transaction and a
   read-only role.

The guard is the user-facing boundary; the engine is the guarantee. Rule: E-04
(prompts are requests, code is the guarantee).

### D-06 — `get_query_error_details` is a real second capability

`execute_sql` returns a short structured error (`error_id`, kind, one-line
message, position). `get_query_error_details(error_id)` returns the full driver
payload plus the column lists of the tables named in the failing SQL, which is
what actually helps the model repair a query. Error records live in a bounded
ring buffer inside the injected database session, so the tool does not read task
state. Rule: T-02, T-05.

### D-07 — Mock data is anchored to a configurable reference date, with a partial current month

`REFERENCE_DATE` defaults to today and pins generation; evaluations set it
explicitly so results stay reproducible. History runs through yesterday, which
means "this month" is month-to-date. The schema digest reports `as_of` so the
planner can compare like with like instead of reporting a fake drop. A naive
answer is a wrong answer here, on purpose.

### D-08 — Dependencies are declared when the phase that uses them lands

`pyproject.toml` lists only `python-dotenv` today. `sqlglot` arrives with phase
2, `openai` with phase 3, `sqlalchemy` + `psycopg[binary]` with phase 7.
Declaring them early would be a dependency that is installed, unversioned in
practice, and unused. Rule: D-01, D-04.

### D-09 — `Settings.from_env(require_llm=False)`

Commands that cannot call a model (`--dry-run`, `--show-schema`, `--query`) skip
only the LLM credential check; every other value is still validated. The
parameter changes behaviour and is honoured; it is not an accepted-and-ignored
knob. Rule: A-07.

### D-10 — `wire.py` is the single composition root

`app/main.py` may not import `db/`, `tools/`, or `agents/` (starter template
section 3), but something must construct them. One module, `wire.py`, knows the
concrete implementations; `main.py` asks it for a database or a schema artifact.
`tests/test_factory.py` parses every module and fails on any other edge, so the
matrix is enforced mechanically rather than by convention. Rule: L-01, M-02.

### D-11 — The mock database is immutable, so its schema is cached

Seeding, introspection, and authorizer installation happen once in
`MockDatabase.build`, in that order. After the authorizer is installed, PRAGMA
and DDL are denied, so introspection could not run anyway; caching the snapshot
is a consequence of immutability, not an optimization. The authorizer denies
writes, DDL, PRAGMA, ATTACH/DETACH, VACUUM-adjacent actions, and explicit
transactions at engine level, which is the guarantee the phase-2 SQL guard sits
on top of. Rule: G-04, D-05.

### D-12 — The planted story is structural, not probabilistic

The drop month's shortfall removes every Electronics item in the North and every
second one elsewhere, by position in the generation stream. A probabilistic skip
made the total revenue change swing between −23% and +10% across seeds, because
one month's SKU mix is noisy. Positional removal gives a stable, attributable
effect on every seed, and North Electronics revenue is *exactly* zero in the drop
month — a fact an evaluation can assert without tolerances. Verified over seeds
1-30. Rule: V-01.

### D-13 — Adapter shapes live in `db/base.py`, not `core/types.py`

`SchemaSnapshot`, `TableInfo`, `QueryRecord`-style adapter types describe an
external system and are shared by the mock and PostgreSQL adapters. The seven
orchestration contracts stay in `core/types.py`; putting backend shapes there
would make the orchestrator aware of database internals. Rule: C-01 (scope),
L-01.

### D-14 — The guard proves read-only; the engine also enforces it

Two independent layers, neither of them a prompt instruction. `db/guard.py` parses
the statement with sqlglot and fails closed: anything it cannot prove is a single
read-only query is refused. The SQLite authorizer in `db/mock_source.py` denies
writes, DDL, PRAGMA, ATTACH, and transactions at engine level. The second layer is
tested directly — `database.execute("DELETE FROM orders", ...)` must return a
`forbidden` error even though it never touched the guard. Rules matter here
because a bypass corpus is only as good as the assumption that no other path
reaches the engine.

The rules are ordered (statement count → statement kind → tree witnesses →
functions) and applied to the parsed tree, never to strings. `WITH x AS (DELETE
FROM orders RETURNING *) SELECT * FROM x` parses as a plain `Select`, which is why
the tree walk exists. Rule: G-04, D-05.

### D-15 — The guard never rewrites SQL

The statement that executes is byte-for-byte the statement the model wrote, minus
a trailing semicolon. An earlier version injected `LIMIT max_rows + 1` into the
tree and re-rendered it; sqlglot's SQLite generator silently dropped CTE column
lists, so `WITH RECURSIVE c(x) AS (...)` became `WITH RECURSIVE c AS (...)`, and a
valid query failed with `no such column: x`. That is a correctness bug in a
component whose only job is safety, and it also broke the audit property that the
SQL in the evidence record is the SQL that ran.

Consequences, accepted deliberately:

- The row cap is enforced at the fetch boundary (`fetchmany(max_rows + 1)`, so
  truncation is detectable and reported), not by editing the query.
- The timeout is a SQLite progress handler, so a pathological sort is stopped by
  time rather than by a LIMIT.
- `MAX_SQL_LENGTH` bounds the input, and `guard` refuses anything larger.

### D-16 — Failures travel as ids, and the journal belongs to the adapter

`execute_sql` returns a short message plus an `error_id`; `get_query_error_details`
exchanges that id for the driver message, the failing SQL, and the columns of
every referenced table, which is the part that actually lets a model repair a
query. The journal lives in the adapter (`ErrorJournal`, bounded at 20) because it
describes the external system, not task state, and because the tool layer stays
stateless.

Consequence: error ids are only meaningful within one process. That is correct for
the agent loop, and it is why the CLI's `--explain` makes the second call in the
same run instead of promising a later lookup. A query that returns zero rows is a
success and is reported as a finding, never as an error. Rule: T-03, D-06.

### D-17 — The client owns attempts; the SDK's retry is disabled

`max_retries=0` is passed to the SDK and the loop lives in
`llm/openai_client.py`, because an attempt the application cannot see is an
attempt it cannot trace, bound, or explain. Retries are spent only on failures
that can plausibly succeed on a second attempt (rate limit, timeout, connection,
5xx); auth, bad-request, and not-found failures fail once and say why. Every
response and error carries `attempts`, so a trace can distinguish "slow" from
"retried twice and still failed".

The request shape also carries tool calls, and the client parses them, even though
phase 3's planner only asks for JSON. Both call modes are required by the design
(tool calling for the investigator, JSON for models that lack it), and the wire format
is the part that is expensive to get wrong: it is tested once here, offline, so phase
4 adds a caller rather than a second serialization.

### D-18 — The planner's JSON is a proposal; validation is code, and repair is once

The model decides the metric, the dates, and the checks. It does not decide the
shape: `parse_plan_payload` checks types, real calendar dates, ordering,
granularity, check count, unique slug ids, and the consistency rules (an
answerable plan needs a check; an unanswerable one needs a gap and no checks), and
it collects every problem instead of failing at the first.

One repair turn, then failure. Each attempt costs latency and money, and a model
that cannot satisfy a stated contract twice will not satisfy it on the fifth try;
a plausible plan that silently violates the contract is worse than an explicit
`plan_invalid`.

Not policed: SQL text appearing in a check's `method`. Detecting it needs a parser
and produces false positives on ordinary English ("select the orders from
October"), while the investigator writes and guards its own SQL anyway, so the
risk is nil and the rule would only cost repairs.

### D-19 — `--plan` runs the real first step, and `--dry-run` stays offline

Phase 3 exposes one model-backed command, and it is the workflow's actual first
step: `select_steps(build_plan(...), ("plan",))` keeps the single source of truth
for the workflow, so the context budget, the trace event, the artifact id, and the
step instruction are identical to what a full run will use in phase 4. A separate
"planner demo" path would have drifted from the real one within two phases.

`--dry-run` remains the offline command (validate configuration, print the plan,
call nothing), so the two are mutually exclusive rather than subtly ordered.

### D-20 — No sampling parameters; capability flags instead

The client sends no `temperature`, `top_p`, or `seed`. Providers disagree about
these (reasoning models reject non-default temperature), and a 400 from a provider
is a worse failure mode than sampling variance. Determinism is pursued where it is
achievable: fixed prompts, structural validation, and pinned fixtures in phase 8.
`LLM_JSON_MODE` exists for the one capability that changes the request body and
that some endpoints reject (`response_format`).

### D-21 - Tool dispatch owns the budgets, and refusals are values

`tools/registry.py` is one entry point for the three tools. It owns the argument
handling, the two budgets, and the evidence records, because that is where the
counters live. `Toolset.call` never raises: an unknown tool, unreadable arguments,
a missing argument, and an exhausted budget all come back as a rendered failure the
model can read. A loop that dies on a malformed call teaches the model nothing.

The budgets are split because they bound different waste:

- `MAX_QUERIES` counts statements that reached the database. A guard refusal never
  touches the database, so charging it here would let a model burn its evidence
  budget on writes; it is charged to the failure allowance instead.
- `MAX_QUERY_RETRIES` allows one attempt plus that many retries before SQL is
  refused outright. This is what stops a model from resending the same broken
  statement eight times and calling it an investigation.

A refusal is recorded as a `QueryRecord` with `error_kind="not_run"` and the SQL
the model wanted to run. The next step needs to know the investigation was cut
short, and a log that silently omits refusals cannot say that.

### D-22 - Findings are structured claims checked against the query log

The investigator does not return prose. It returns one JSON object of
`findings`, each with an id, a claim, the `query_id` values that support it, and
the numbers it asserts. `parse_findings_payload` accepts an evidence id only if a
query with that id returned rows, and rejects a claim with no evidence, an unknown
id, a non-numeric number, a duplicate id, or an empty report. Every problem is
collected and sent back once, the same contract as the planner (D-18).

This buys one real invariant: a claim cannot cite evidence that does not exist.
It does not buy arithmetic: a claim may cite a real query and misquote its rows.
That needs a verifier, which is what phase 8's golden evaluations are for. To make
that possible, every record keeps its `query_id`, the SQL that actually ran, and a
bounded preview of the rows (added to `QueryRecord` for this reason; `preview` is
additive and `from_dict` tolerates its absence in older payloads).

The alternative - trusting prose and asking the model to be careful - makes the
project's central claim untestable.

### D-23 - A tool's spec belongs to the tool; reading JSON belongs to the agents

Two consequences of building the loop, both recorded as deviations:

- `tools/registry.py` builds `ToolSpec`s, which live in `llm/base.py`, so `tools`
  now depends on `llm`. The direction is one-way (a tool describes itself for a
  model; the model layer never imports tools) and acyclic, and the alternative -
  moving `ToolSpec` into `core/types.py` - scatters the model boundary for no
  behavioural gain. The import matrix in `tests/test_factory.py` was widened to
  say so rather than bypassed.
- Both agents read a JSON reply and reject it the same way, so the reading and the
  repair wording moved to `agents/replies.py`. Each agent keeps its own rejection
  type (`PlanPayloadInvalid`, `FindingsInvalid`) and its own validator; only the
  extraction and the sentence a model reads are shared.

A message with no tool calls is the investigator's natural stop, so the findings
JSON arrives in the same turn as the last tool batch; `json_mode` is switched on
only for the repair turn and the forced final turn, where no tools are offered,
because some providers reject `response_format` together with tools.

### D-24 - Tool calling is a capability flag, and the fallback is an adapter

`LLM_SUPPORTS_TOOL_CALLING` (default `true`) decides whether the investigator's
tool loop uses native `tool_calls` or the structured-JSON fallback. The fallback
is `llm/structured_client.py`, a transparent `LLMClient` adapter: it rewrites a
tool request into a JSON-schema request and parses a `tool_calls` array out of
the reply. Agents never change, so the planner, the reporter, and the
investigator's message shape, budgets, repair turn, and findings turn are all
identical in both modes.

The adapter is transparent for requests that carry no tools, which is every
request except the investigator's tool turns. A reply with no `tool_calls` key is
the model's findings and returns an empty tuple, so the investigator's stop
condition works unchanged; invalid JSON also returns an empty tuple, which lets
the investigator's single repair turn ask for the JSON it actually needs.

This is the D-17 pattern again: the wire format is tested once, offline, so phase
6 adds a caller rather than a second serialization. Rule D-17, D-23.

---

## Deviations log

```
Rule: A-01 / starter template "never skip T1 to start at T2"
Deviation: core/ is split into types, config, context, trace, orchestrator from day one
Reason: ten contracts plus six domain payloads do not fit one module; no store, no worker,
        no transport abstraction has been added. Reviewed 2026-11-17.
```

```
Rule: starter template section 3 (tools must not import the database layer)
Deviation: tools/sql.py imports db/
Reason: the database is the external system, not task state; task state stays in the
        orchestrator. See D-04. Reviewed 2026-11-17.
```

```
Rule: C-01 (contracts are minimal dataclasses per Appendix A)
Deviation: AgentResult gains a `data` field
Reason: structured plan and report payloads are consumed by code, not prose. Additive and
        JSON-serializable. See D-03. Reviewed 2026-11-17.
```

```
Rule: S-04 (persistence tier)
Deviation: no persistence at all; TaskState lives in memory for the duration of one run
Reason: single-run CLI, no resume requirement. The trace file is the durable artifact.
        Revisit if runs become resumable. Reviewed 2026-11-17.
```

```
Rule: starter template section 3 (app/main.py must not import tools or db)
Deviation: app/main.py imports wire.py, which imports them
Reason: a composition root has to exist somewhere; keeping it in one module preserves the
        rule that main knows no concrete capability. Enforced by an import-graph test.
        Reviewed 2026-11-17.
```

```
Rule: C-01 (contracts belong in core/types.py)
Deviation: db/base.py defines its own adapter shapes
Reason: those types describe an external system, not orchestration state; sharing them in
        core would leak database vocabulary into the orchestrator. See D-13. Reviewed 2026-11-17.
```

```
Rule: C-01 (contracts belong in core/types.py)
Deviation: llm/base.py defines the model boundary, and agents/planner.py defines
           PlanPayloadInvalid
Reason: same class as db/base.py - messages, tool calls, and client failures describe an
        external system, and the validation error is private to one agent's contract. A
        second consumer (phase 4's JSON fallback) imports llm/base.py rather than duplicating
        it. Reviewed 2026-11-17.
```

```
Rule: starter template section 3 (capabilities do not import each other)
Deviation: tools/registry.py imports llm/base.py for ToolSpec and ToolCall
Reason: a tool's spec is its model-facing description, and the spec shape lives at the model
        boundary. The dependency is one-way (llm never imports tools) and acyclic. The import
        matrix test was widened to allow it rather than bypassed. See D-23. Reviewed 2026-11-17.
```

```
Rule: C-01 (contracts are the shapes in core/types.py)
Deviation: QueryRecord gains `preview`; ToolError gains `error_id`
Reason: both are additive and JSON-serializable, and from_dict tolerates their absence. The
        report needs a bounded row excerpt to cite numbers without re-running queries, and the
        registry needs the id without parsing the rendering. See D-21, D-22. Reviewed 2026-11-17.
```


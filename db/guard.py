"""SQL validation: the security boundary is code, not a prompt instruction.

Rules are enforced on the parsed syntax tree, never by string matching, and the
guard fails closed: anything it cannot prove is a single read-only query is
rejected. The database engine's own read-only mode sits underneath as a second,
independent layer (see `db/mock_source.py`).

Order of checks matters: statement count, then statement kind, then the tree
witnesses (SELECT INTO, locks, embedded DML, dangerous functions). A `WITH`
clause is the reason the tree walk exists: sqlglot parses
`WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x` as a plain `Select`.

The guard reads the tree but never writes it back. Re-rendering parsed SQL is
lossy (sqlglot's SQLite generator drops CTE column lists, for instance), so the
statement that executes is byte-for-byte the statement the model wrote, and the
row cap is enforced when fetching instead of by injecting a LIMIT.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

# Backend name -> sqlglot dialect.
BACKEND_DIALECTS: dict[str, str] = {"mock": "sqlite", "postgres": "postgres"}

# Statement-level nodes that must never appear, at the top level or nested in a
# CTE. Collected defensively: sqlglot adds and removes node classes between
# releases, and a missing name must not break the guard at import time.
_FORBIDDEN_NODE_NAMES: tuple[str, ...] = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Create",
    "Drop",
    "Alter",
    "TruncateTable",
    "Command",
    "Transaction",
    "Commit",
    "Rollback",
    "Attach",
    "Detach",
    "Pragma",
    "Set",
    "Grant",
    "Revoke",
    "Analyze",
    "Describe",
    "Copy",
    "Use",
    "LoadData",
    "Kill",
    "Cache",
    "Uncache",
    "Refresh",
)
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    node
    for node in (getattr(exp, name, None) for name in _FORBIDDEN_NODE_NAMES)
    if isinstance(node, type) and issubclass(node, exp.Expression)
)

# Functions that touch the filesystem, the network, the server, or sequences.
# Everything else (aggregates, window functions, string/date functions) is fine.
_FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        # SQLite extensions
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        # PostgreSQL: server, filesystem, network
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_export_snapshot",
        "pg_logical_emit_message",
        "pg_create_restore_point",
        "pg_switch_wal",
        "pg_start_backup",
        "pg_stop_backup",
        "pg_advisory_lock",
        "pg_advisory_xact_lock",
        "pg_try_advisory_lock",
        "pg_try_advisory_xact_lock",
        "dblink",
        "dblink_connect",
        "dblink_exec",
        "lo_import",
        "lo_export",
        "lo_unlink",
        "lo_create",
        "lo_creat",
        "set_config",
        "current_setting",
        "nextval",
        "setval",
        "query_to_xml",
        "query_to_xmlschema",
        "cursor_to_xml",
        "table_to_xml",
        "database_to_xml",
    }
)

_HEAD_ACCEPTED_HINT = (
    "send one read-only SELECT (a WITH ... SELECT is fine); the database is queried, never changed"
)


@dataclass(frozen=True)
class GuardRejection:
    """Why the SQL was refused. `rule` is stable and machine-readable."""

    rule: str
    message: str
    hint: str = _HEAD_ACCEPTED_HINT


@dataclass(frozen=True)
class GuardedQuery:
    """SQL that has been proven read-only.

    `sql` is the statement exactly as the model wrote it (only the trailing
    semicolon removed). The guard never rewrites SQL: rendering the parsed tree
    back to text is lossy enough to change meaning, so the row cap is enforced
    when fetching, not by editing the query.
    """

    sql: str
    tables: tuple[str, ...]


def guard(
    sql: str,
    *,
    dialect: str,
    max_length: int = 20_000,
) -> GuardedQuery | GuardRejection:
    """Prove `sql` is one read-only query.

    Returns either the statement to execute (verbatim) or a rejection. Never
    raises on malformed input.
    """
    text = (sql or "").strip()
    if text.endswith(";"):
        text = text[:-1].strip()

    if not text:
        return GuardRejection(rule="empty_sql", message="no SQL was provided")

    if len(text) > max_length:
        return GuardRejection(
            rule="too_long",
            message=f"the statement is {len(text)} characters; the limit is {max_length}",
            hint="select fewer columns or simplify the query",
        )

    try:
        with _quiet_sqlglot():
            statements = [s for s in sqlglot.parse(text, read=dialect) if s is not None]
    except Exception as exc:  # sqlglot raises ParseError and friends
        return GuardRejection(
            rule="parse_error",
            message=f"the SQL could not be parsed for dialect {dialect}: {_first_line(exc)}",
            hint="check table names, column names, and quoting against the schema",
        )

    if not statements:
        return GuardRejection(rule="empty_sql", message="no SQL was provided")

    if len(statements) > 1:
        return GuardRejection(
            rule="multi_statement",
            message=f"{len(statements)} statements were provided; exactly one is allowed",
        )

    expression = statements[0]

    if not isinstance(expression, exp.Query):
        return GuardRejection(
            rule="not_select",
            message=(
                f"only read-only queries are allowed; this is a {type(expression).__name__} statement"
            ),
        )

    if expression.args.get("into"):
        return GuardRejection(
            rule="select_into",
            message="SELECT ... INTO writes a new table and is not allowed",
            hint="use a plain SELECT; results come back as rows",
        )

    if expression.args.get("locks"):
        return GuardRejection(
            rule="locking_read",
            message="row-locking clauses such as FOR UPDATE are not allowed",
            hint="remove the locking clause; this database is never modified",
        )

    forbidden = _find_forbidden_node(expression)
    if forbidden is not None:
        return GuardRejection(
            rule="forbidden_statement",
            message=(
                f"the query contains an embedded {forbidden} statement, which is not allowed"
                if isinstance(expression, exp.Select)
                else f"{forbidden} statements are not allowed"
            ),
            hint="data-modifying statements are refused even inside a WITH clause",
        )

    function = _find_forbidden_function(expression)
    if function is not None:
        return GuardRejection(
            rule="forbidden_function",
            message=f"the function {function}() is not allowed",
            hint="use standard SQL aggregates and window functions",
        )

    return GuardedQuery(sql=text, tables=_referenced_tables(expression))


def _find_forbidden_node(expression: exp.Expression) -> str | None:
    for node in expression.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return type(node).__name__
    return None


def _find_forbidden_function(expression: exp.Expression) -> str | None:
    for node in expression.find_all(exp.Func):
        name = _function_name(node)
        if name in _FORBIDDEN_FUNCTIONS:
            return name
    return None


def _function_name(node: exp.Func) -> str:
    """`exp.Anonymous` keeps the written name; typed functions expose sql_name()."""
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return (node.sql_name() or "").lower()


def _referenced_tables(expression: exp.Expression) -> tuple[str, ...]:
    names: list[str] = []
    for table in expression.find_all(exp.Table):
        if table.name and table.name not in names:
            names.append(table.name)
    return tuple(names)


def _first_line(exc: BaseException) -> str:
    message = str(exc).strip().replace("\n", " ")
    return message[:200] if message else type(exc).__name__


@contextmanager
def _quiet_sqlglot() -> Iterator[None]:
    """sqlglot warns before falling back to a generic node for unsupported syntax.

    We reject those statements with a clearer message, so the warning would only
    duplicate it. Scoped to the parse call and restored afterwards.
    """
    logger = logging.getLogger("sqlglot")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)

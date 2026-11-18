"""The guard is a security boundary, so it is tested adversarially.

Every case here is a bypass attempt that must be refused, plus legitimate
analytics SQL that must pass untouched. Rules are asserted by name, not by
message text, so wording can change without weakening the tests.
"""

from __future__ import annotations

import pytest

from db.guard import (
    _FORBIDDEN_NODES,
    GuardRejection,
    GuardedQuery,
    guard,
)

MAX_ROWS = 500


def run(sql: str, *, dialect: str = "sqlite", max_length: int = 20_000):
    return guard(sql, dialect=dialect, max_length=max_length)


def rule_of(sql: str, **kwargs) -> str:
    outcome = run(sql, **kwargs)
    assert isinstance(outcome, GuardRejection), f"accepted: {sql!r} -> {outcome!r}"
    return outcome.rule


# --------------------------------------------------------------------------
# Statement count and kind
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE orders",
        "SELECT 1; SELECT 2",
        "SELECT * FROM orders; DELETE FROM orders",
        "SELECT 1; DROP TABLE orders; SELECT 2",
    ],
)
def test_multiple_statements_are_refused(sql: str) -> None:
    assert rule_of(sql) == "multi_statement"


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE orders SET status = 'x'",
        "INSERT INTO customers (id) VALUES (1)",
        "CREATE TABLE x (id int)",
        "CREATE VIEW v AS SELECT 1",
        "ALTER TABLE orders ADD COLUMN x int",
        "TRUNCATE TABLE orders",
        "ATTACH DATABASE 'x.db' AS x",
        "DETACH DATABASE x",
        "PRAGMA table_info(orders)",
        "VACUUM",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "COPY order_items FROM 'x.csv'",
        "MERGE INTO t USING u ON 1 = 1 WHEN MATCHED THEN DELETE",
        "GRANT ALL ON orders TO analyst",
        "REVOKE ALL ON orders FROM analyst",
        "SET search_path = evil",
        "EXPLAIN SELECT 1",
        "ANALYZE",
        "USE other_database",
    ],
)
def test_writes_ddl_and_admin_statements_are_refused(sql: str) -> None:
    assert rule_of(sql) == "not_select"


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
        "WITH x AS (UPDATE orders SET status = 'x' RETURNING *) SELECT * FROM x",
        "WITH x AS (INSERT INTO customers (id) VALUES (1) RETURNING *) SELECT * FROM x",
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x UNION SELECT 1",
        "SELECT * FROM (WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x)",
        "SELECT 1 WHERE 1 IN (WITH x AS (UPDATE orders SET a = 1 RETURNING *) SELECT 1 FROM x)",
        "WITH x AS (INSERT INTO orders (id) VALUES (1) RETURNING *) SELECT * FROM x ORDER BY 1",
    ],
)
def test_data_modifying_ctes_are_refused(sql: str) -> None:
    """sqlglot parses these as a plain Select, so the tree walk is mandatory."""
    assert rule_of(sql) == "forbidden_statement"


def test_a_cte_that_sqlglot_cannot_parse_is_still_refused() -> None:
    outcome = run("WITH a AS (SELECT 1), b AS (DROP TABLE orders) SELECT * FROM a")
    assert isinstance(outcome, GuardRejection)
    assert outcome.rule in {"forbidden_statement", "parse_error"}


def test_select_into_is_refused() -> None:
    assert rule_of("SELECT * INTO new_table FROM orders") == "select_into"


def test_row_locking_reads_are_refused() -> None:
    assert rule_of("SELECT * FROM orders FOR UPDATE") == "locking_read"


# --------------------------------------------------------------------------
# Dangerous functions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT load_extension('evil')",
        "SELECT LOAD_EXTENSION('evil')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/x', 'y')",
        "SELECT pg_sleep(10)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_read_binary_file('/etc/passwd')",
        "SELECT dblink('host=1', 'select 1')",
        "SELECT dblink_exec('host=1', 'delete from t')",
        "SELECT lo_import('/etc/passwd')",
        "SELECT lo_export(1, '/tmp/x')",
        "SELECT set_config('a', 'b', false)",
        "SELECT current_setting('data_directory')",
        "SELECT nextval('orders_id_seq')",
        "SELECT setval('orders_id_seq', 1)",
        "SELECT query_to_xml('delete from t', true, true, '')",
    ],
)
def test_dangerous_functions_are_refused(sql: str) -> None:
    assert rule_of(sql) == "forbidden_function"


def test_dangerous_function_nested_in_an_expression_is_refused() -> None:
    assert rule_of("SELECT 1 + (SELECT count(*) FROM t WHERE pg_sleep(2) IS NULL)") == (
        "forbidden_function"
    )


# --------------------------------------------------------------------------
# Malformed and oversized input
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sql", ["", "   ", "\n\t ", ";", "-- only a comment", "/* x */"])
def test_empty_input_is_refused(sql: str) -> None:
    assert rule_of(sql) == "empty_sql"


@pytest.mark.parametrize("sql", ["not sql at all", "SELECT FROM WHERE", "SELECT * FROM"])
def test_unparseable_input_is_refused(sql: str) -> None:
    assert rule_of(sql) == "parse_error"


def test_oversized_sql_is_refused() -> None:
    assert rule_of("SELECT " + "'x' , " * 5_000 + "1") == "too_long"


def test_rejections_never_raise() -> None:
    hostile = ["SELECT \x00", "select " + "(" * 500 + "1" + ")" * 500, "😀"]
    for sql in hostile:
        outcome = run(sql)
        assert isinstance(outcome, (GuardedQuery, GuardRejection))


# --------------------------------------------------------------------------
# Legitimate SQL must pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT COUNT(*) FROM orders",
        "SELECT status, SUM(i.line_total) FROM orders o "
        "JOIN order_items i ON i.order_id = o.id GROUP BY status HAVING SUM(i.line_total) > 0",
        "WITH monthly AS (SELECT 1 AS m) SELECT m FROM monthly",
        "SELECT a FROM t UNION ALL SELECT b FROM u",
        "SELECT RANK() OVER (PARTITION BY x ORDER BY y DESC) FROM t",
        "SELECT * FROM (SELECT 1 AS inner_value) nested",
        "SELECT date(order_date, 'start of month') FROM orders",
        "SELECT '; DROP TABLE orders'",
        "SELECT 1 -- ; DROP TABLE orders",
        "SELECT 1;",
        "  SELECT 1  ",
        "select * from customers",
        "SELECT json_extract('{\"a\":1}', '$.a')",
        "SELECT cast(1 AS INTEGER) + round(2.5)",
    ],
)
def test_legitimate_queries_are_accepted(sql: str) -> None:
    outcome = run(sql)
    assert isinstance(outcome, GuardedQuery), f"refused: {sql!r} -> {outcome!r}"


def test_semicolon_inside_a_string_literal_is_not_a_statement_separator() -> None:
    outcome = run("SELECT 'a;b' AS value")
    assert isinstance(outcome, GuardedQuery)
    assert "a;b" in outcome.sql


# --------------------------------------------------------------------------
# The statement is executed exactly as written
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "SELECT * FROM orders LIMIT 5",
        "SELECT * FROM orders LIMIT 999999",
        "SELECT * FROM orders LIMIT -1",
        "SELECT * FROM orders LIMIT 0",
        "SELECT * FROM orders LIMIT 5 OFFSET 10",
        "SELECT a FROM t UNION ALL SELECT b FROM u",
        "WITH counter(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM counter WHERE x < 5) "
        "SELECT COUNT(*) FROM counter",
        "SELECT 1 + 1 AS two",
    ],
)
def test_sql_is_never_rewritten(sql: str) -> None:
    """The guard only reads. A rewrite can change meaning; the cap is applied on fetch."""
    outcome = run(sql)
    assert isinstance(outcome, GuardedQuery)
    assert outcome.sql == sql


def test_trailing_semicolon_and_surrounding_space_are_trimmed() -> None:
    outcome = run("  SELECT 1 ;  ")
    assert isinstance(outcome, GuardedQuery)
    assert outcome.sql == "SELECT 1"


def test_cte_column_list_survives_the_guard() -> None:
    """sqlglot's SQLite generator drops named CTE columns; we never re-render."""
    sql = (
        "WITH RECURSIVE counter(x) AS ("
        "SELECT 1 UNION ALL SELECT x + 1 FROM counter WHERE x < 3"
        ") SELECT x FROM counter"
    )
    outcome = run(sql)
    assert isinstance(outcome, GuardedQuery)
    assert "counter(x)" in outcome.sql


def test_referenced_tables_are_collected() -> None:
    outcome = run(
        "SELECT * FROM orders o JOIN customers c ON c.id = o.customer_id "
        "JOIN order_items i ON i.order_id = o.id"
    )
    assert isinstance(outcome, GuardedQuery)
    assert outcome.tables == ("orders", "customers", "order_items")


def test_cte_names_are_reported_alongside_real_tables() -> None:
    outcome = run("WITH m AS (SELECT * FROM orders) SELECT * FROM m")
    assert isinstance(outcome, GuardedQuery)
    assert "orders" in outcome.tables


def test_query_without_tables_reports_none() -> None:
    outcome = run("SELECT 1")
    assert isinstance(outcome, GuardedQuery)
    assert outcome.tables == ()


def test_trailing_comment_cannot_smuggle_a_statement() -> None:
    outcome = run("SELECT 1 -- ; DROP TABLE orders")
    assert isinstance(outcome, GuardedQuery)
    assert "DROP" in outcome.sql  # inside the comment, and the guard proved that


# --------------------------------------------------------------------------
# Dialects and internal wiring
# --------------------------------------------------------------------------


def test_guard_works_for_the_postgres_dialect() -> None:
    outcome = run("SELECT * FROM orders LIMIT 10", dialect="postgres")
    assert isinstance(outcome, GuardedQuery)
    assert "LIMIT 10" in outcome.sql.upper()


def test_select_into_is_caught_in_the_postgres_dialect() -> None:
    assert rule_of("SELECT * INTO new_t FROM orders", dialect="postgres") == "select_into"


def test_postgres_copy_is_refused() -> None:
    assert rule_of("COPY t FROM '/etc/passwd'", dialect="postgres") == "not_select"


def test_forbidden_node_table_covers_the_dangerous_statements() -> None:
    """Guards against a sqlglot rename silently disabling a rule."""
    names = {node.__name__ for node in _FORBIDDEN_NODES}
    assert {
        "Insert",
        "Update",
        "Delete",
        "Create",
        "Drop",
        "Alter",
        "Command",
        "Pragma",
        "Transaction",
        "Attach",
    } <= names

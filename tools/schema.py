"""Schema tools: render schema facts as text a model can act on.

Two shapes, one purpose:
- a bounded digest for the planner, truncated at table boundaries;
- `inspect_schema(table)`, a tool for detail on demand.

The tool signature and docstring are the schema the model sees (T-01, T-06).
"""

from __future__ import annotations

from collections.abc import Callable

from core.types import ToolError
from db.base import Database, SchemaSnapshot, TableDescription, TableInfo

InspectSchemaTool = Callable[[str], str | ToolError]

_VALUE_WIDTH = 40


def render_schema_digest(snapshot: SchemaSnapshot, budget: int) -> tuple[str, bool]:
    """Compact schema, truncated at table boundaries so no column line is cut.

    Returns `(text, truncated)`. `text` never exceeds `budget`.
    """
    header = [
        f"DATABASE SCHEMA (backend: {snapshot.backend}, as of: {snapshot.as_of}, "
        f"tables: {len(snapshot.tables)})",
    ]
    if snapshot.data_span is not None:
        header.append(f"data span: {snapshot.data_span[0]} .. {snapshot.data_span[1]}")
    for note in snapshot.notes:
        header.append(f"note: {note}")
    header.append("use inspect_schema(table) for sample rows and exact value formats")

    head = "\n".join(header)
    if budget <= 0:
        return "", True
    if len(head) > budget:
        return head[:budget], True

    chunks = [head]
    used = len(head)
    omitted: list[str] = []

    for table in snapshot.tables:
        for candidate in (_full_block(table), _compact_block(table)):
            cost = 2 + len(candidate)
            if used + cost <= budget:
                chunks.append(candidate)
                used += cost
                break
        else:
            omitted.append(table.name)

    if not omitted:
        return "\n\n".join(chunks), False

    text = "\n\n".join(chunks)
    footer = (
        f"\n\n[truncated: {len(omitted)} of {len(snapshot.tables)} tables omitted "
        f"to fit the budget: {', '.join(omitted)}]"
    )
    if len(text) + len(footer) <= budget:
        return text + footer, True
    short_footer = "\n\n[truncated]"
    if len(text) + len(short_footer) <= budget:
        return text + short_footer, True
    return text[:budget], True


def render_table_detail(description: TableDescription) -> str:
    """Everything needed to write correct SQL against one table."""
    table = description.table
    lines = [_full_block(table), "", "value domains:"]
    if description.distinct_values:
        width = max(len(values.column) for values in description.distinct_values)
        lines.extend(
            f"  {values.column.ljust(width)}  {', '.join(values.values)}"
            for values in description.distinct_values
        )
    else:
        lines.append("  (no low-cardinality columns)")

    lines.extend(["", f"sample rows ({len(description.sample_rows)}):"])
    if description.sample_rows:
        for row in description.sample_rows:
            lines.append(
                "  " + " | ".join(f"{key}={_short(value)}" for key, value in row.items())
            )
    else:
        lines.append("  (table is empty)")
    return "\n".join(lines)


def make_inspect_schema_tool(database: Database) -> InspectSchemaTool:
    """Bind the schema capability to one database. The returned function is the tool."""

    def inspect_schema(table: str) -> str | ToolError:
        """Return the full definition of one table: columns with types and keys,
        the actual stored values of low-cardinality columns, and sample rows.

        Use this before writing SQL against a table you have not inspected.

        Args:
            table: exact table name from the schema digest, e.g. "orders".

        Returns:
            Plain text. On success: the table block, "value domains" listing the
            real values of categorical columns, and up to 3 sample rows.
            On failure: a ToolError with kind "invalid_input" when the table does
            not exist, or "internal" when the database read fails.
        """
        name = (table or "").strip().strip('"')
        snapshot = database.schema()
        info = snapshot.table(name)
        if not name:
            return ToolError(
                tool="inspect_schema",
                kind="invalid_input",
                message="table is required",
                detail=f"known tables: {', '.join(snapshot.table_names)}",
            )
        if info is None:
            return ToolError(
                tool="inspect_schema",
                kind="invalid_input",
                message=f"unknown table {name!r}",
                detail=f"known tables: {', '.join(snapshot.table_names)}",
            )
        try:
            description = database.describe(info.name)
        except Exception as exc:  # noqa: BLE001 - recoverable for the loop (T-03)
            return ToolError(
                tool="inspect_schema",
                kind="internal",
                message=f"{type(exc).__name__}: {exc}",
                retryable=False,
            )
        return render_table_detail(description)

    return inspect_schema


def _full_block(table: TableInfo) -> str:
    lines = [f"{table.name}  ({table.row_count} rows)"]
    width = max((len(column.name) for column in table.columns), default=0)
    foreign_keys = {fk.column: fk for fk in table.foreign_keys}
    for column in table.columns:
        flags: list[str] = []
        if column.primary_key:
            flags.append("PK")
        elif not column.nullable:
            flags.append("NOT NULL")
        foreign_key = foreign_keys.get(column.name)
        if foreign_key is not None:
            flags.append(
                f"-> {foreign_key.references_table}.{foreign_key.references_column}"
            )
        suffix = f"  {', '.join(flags)}" if flags else ""
        lines.append(f"  {column.name.ljust(width)}  {column.type}{suffix}")
    return "\n".join(lines)


def _compact_block(table: TableInfo) -> str:
    columns = ", ".join(column.name for column in table.columns)
    return f"{table.name} ({table.row_count} rows): {columns}"


def _short(value: object, limit: int = _VALUE_WIDTH) -> str:
    if value is None:
        return "NULL"
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"

"""Backend selection and the architectural dependency rules.

The import-graph test is static (AST only): it never imports the modules it
checks, so it also guards code that would fail to run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.config import Settings
from core.orchestrator import AGENT_KEYS
from core.types import BackendUnavailable
from db.base import Database
from db.factory import build_database
from db.mock_source import MockDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Which top-level package each package is allowed to import from.
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "app": {"app", "core", "wire"},
    "core": {"core"},
    "db": {"db", "core"},
    "tools": {"tools", "db", "core", "llm"},
    "agents": {"agents", "core", "tools", "llm", "db"},
    "llm": {"llm", "core"},
    "evals": {"evals", "core", "wire", "db", "tools", "agents", "llm"},
    "wire": {"wire", "core", "db", "tools", "agents", "llm"},
    "tests": {"tests", "app", "core", "db", "tools", "agents", "llm", "wire"},
}


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------


def test_mock_backend_is_built_from_settings(settings: Settings) -> None:
    database = build_database(settings)
    assert isinstance(database, MockDatabase)
    assert database.backend == "mock"


def test_mock_backend_matches_the_configured_seed(env: dict[str, str]) -> None:
    settings = Settings.from_env({**env, "MOCK_SEED": "11"})
    first = build_database(settings)
    second = build_database(Settings.from_env({**env, "MOCK_SEED": "11"}))
    assert first.schema().table("orders").row_count == second.schema().table("orders").row_count


def test_postgres_backend_connects_and_fails_closed_on_bad_url(
    env: dict[str, str],
) -> None:
    """The PostgreSQL adapter connects at build time; a bad URL is a typed failure.

    Unlike the mock, which never touches the network, the real backend opens a
    connection during construction so a misconfiguration surfaces before the
    first query rather than mid-investigation.
    """
    settings = Settings.from_env(
        {
            **env,
            "DATABASE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql+psycopg://analyst:pw@127.0.0.1:1/analytics",
        }
    )
    with pytest.raises(BackendUnavailable) as caught:
        build_database(settings)
    assert "could not connect" in str(caught.value).lower() or "connection" in str(caught.value).lower()


def test_built_database_satisfies_the_protocol(settings: Settings) -> None:
    database: Database = build_database(settings)
    assert database.schema().table_names
    assert database.describe("orders").table.name == "orders"


# --------------------------------------------------------------------------
# Agent composition
# --------------------------------------------------------------------------


def test_wire_exposes_only_agents_that_exist(settings: Settings) -> None:
    """The workflow declares three roles; this build implements a subset of them."""
    import wire

    databases = build_database(settings)
    agents = wire.build_agents(settings, databases)
    assert set(agents) <= set(AGENT_KEYS)
    assert {"planner", "investigator"} <= set(agents)
    assert callable(agents["planner"])
    assert callable(agents["investigator"])


def test_wire_builds_a_model_client_from_settings(settings: Settings) -> None:
    import wire
    from llm.base import LLMClient

    client: LLMClient = wire.build_llm(settings)
    assert callable(client.complete)


# --------------------------------------------------------------------------
# Import graph
# --------------------------------------------------------------------------


def python_files() -> list[Path]:
    return [
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if not any(part.startswith(".") or part == "__pycache__" for part in path.parts)
    ]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import stays inside its own package
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_every_module_respects_the_layer_matrix() -> None:
    violations: list[str] = []
    for path in python_files():
        relative = path.relative_to(PROJECT_ROOT)
        package = relative.parts[0]
        if package not in ALLOWED_IMPORTS:
            continue
        allowed = ALLOWED_IMPORTS[package]
        for root in imported_roots(path):
            if root in ALLOWED_IMPORTS and root not in allowed:
                violations.append(f"{relative}: {package}/ may not import {root}/")
    assert not violations, "\n".join(violations)


def test_core_does_not_import_capabilities() -> None:
    for path in (PROJECT_ROOT / "core").glob("*.py"):
        forbidden = imported_roots(path) & {"db", "tools", "agents", "app", "llm"}
        assert not forbidden, f"{path.name} imports {forbidden}"


def test_capabilities_do_not_import_the_orchestrator() -> None:
    for package in ("db", "tools", "llm"):
        for path in (PROJECT_ROOT / package).glob("*.py"):
            module = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(module):
                if isinstance(node, ast.ImportFrom) and node.module == "core.orchestrator":
                    raise AssertionError(f"{path} imports the orchestrator")


def test_only_the_composition_root_builds_adapters() -> None:
    """Anything that constructs a database goes through wire/db.factory.

    Tests are exempt: building one in a fixture is the point of a test double, and
    they never ship.
    """
    builders = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in python_files()
        if "build_database(" in path.read_text(encoding="utf-8")
    }
    shipping = {path for path in builders if not path.startswith("tests/")}
    assert shipping <= {"wire.py", "db/factory.py"}


def test_the_import_graph_test_is_not_vacuous() -> None:
    """Guard against a typo that turns the matrix check into a no-op."""
    scanned = {path.relative_to(PROJECT_ROOT).parts[0] for path in python_files()}
    assert {"core", "app", "db", "tools"} <= scanned
    assert imported_roots(PROJECT_ROOT / "tools" / "schema.py") >= {"core", "db"}

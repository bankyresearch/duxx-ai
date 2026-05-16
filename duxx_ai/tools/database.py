"""Database domain tools for Duxx AI agents.

Provides tools for executing SQL queries, NoSQL queries, and describing
table schemas. Uses placeholder implementations by default; for real
usage, configure connection strings pointing to your database instances.

Supported backends (planned):
    SQL: PostgreSQL, MySQL, SQLite via appropriate drivers
    NoSQL: MongoDB via pymongo, Redis via redis-py
"""

from __future__ import annotations

import json

from duxx_ai.core.tool import Tool, tool

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="sql_query",
    description="Execute a SQL query against a database. Use with caution -- mutating queries require approval.",
    requires_approval=True,
    tags=["database", "sql"],
)
def sql_query(query: str, connection_string: str = "", params: str = "{}") -> str:
    """Execute a SQL query and return results.

    Args:
        query: The SQL query to execute.
        connection_string: Database connection URI
            (e.g. postgresql://user:pass@host/db, sqlite:///path.db).
            Leave empty to use the default configured connection.
        params: JSON-encoded dict of query parameters for parameterised queries.

    Returns:
        JSON-formatted query results or error message.
    """
    if not query or not query.strip():
        return "Error: SQL query string is required."

    # Parse params
    try:
        param_dict = json.loads(params) if params else {}
    except json.JSONDecodeError as e:
        return f"Error: invalid params JSON -- {e}"

    # Placeholder implementation
    query_lower = query.strip().lower()
    query_type = query_lower.split()[0] if query_lower else "unknown"

    if query_type == "select":
        sample_result = {
            "columns": ["id", "name", "created_at"],
            "rows": [
                [1, "sample_row_1", "2026-01-15T10:00:00"],
                [2, "sample_row_2", "2026-02-20T14:30:00"],
            ],
            "row_count": 2,
        }
        return (
            f"[PLACEHOLDER] Query executed successfully.\n"
            f"  Type: SELECT\n"
            f"  Connection: {connection_string or '(default)'}\n"
            f"  Params: {param_dict}\n\n"
            + json.dumps(sample_result, indent=2)
        )
    elif query_type in ("insert", "update", "delete"):
        return (
            f"[PLACEHOLDER] Mutating query acknowledged.\n"
            f"  Type: {query_type.upper()}\n"
            f"  Connection: {connection_string or '(default)'}\n"
            f"  Affected rows: 1 (placeholder)\n"
            f"\n"
            f"Note: Connect a real database for actual execution."
        )
    else:
        return (
            f"[PLACEHOLDER] Query type '{query_type}' acknowledged.\n"
            f"  Connection: {connection_string or '(default)'}\n"
            f"  Configure a real database driver for execution."
        )


@tool(
    name="nosql_query",
    description="Execute a NoSQL query against a document database.",
    tags=["database", "nosql"],
)
def nosql_query(collection: str, filter: str = "{}", limit: int = 10) -> str:
    """Query a NoSQL / document database collection.

    Args:
        collection: Collection or table name.
        filter: JSON-encoded filter/query document.
        limit: Maximum documents to return.

    Returns:
        JSON-formatted results.
    """
    if not collection or not collection.strip():
        return "Error: collection name is required."
    if limit < 1:
        return "Error: limit must be >= 1."

    try:
        filter_dict = json.loads(filter) if filter else {}
    except json.JSONDecodeError as e:
        return f"Error: invalid filter JSON -- {e}"

    sample_docs = [
        {
            "_id": f"doc_{i}",
            "collection": collection,
            "data": {"key": f"value_{i}", "counter": i * 10},
            "created_at": "2026-03-01T12:00:00Z",
        }
        for i in range(1, min(limit, 3) + 1)
    ]

    return (
        f"[PLACEHOLDER] NoSQL query executed.\n"
        f"  Collection: {collection}\n"
        f"  Filter: {json.dumps(filter_dict)}\n"
        f"  Limit: {limit}\n"
        f"  Returned: {len(sample_docs)} documents\n\n"
        + json.dumps(sample_docs, indent=2)
    )


@tool(
    name="describe_table",
    description="Describe the schema/structure of a database table or collection.",
    tags=["database", "schema"],
)
def describe_table(table_name: str, connection_string: str = "") -> str:
    """Get the schema of a database table.

    Args:
        table_name: Name of the table or collection to describe.
        connection_string: Database connection URI (optional).

    Returns:
        JSON-formatted schema description.
    """
    if not table_name or not table_name.strip():
        return "Error: table_name is required."

    sample_schema = {
        "table": table_name,
        "columns": [
            {"name": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
            {"name": "name", "type": "VARCHAR(255)", "nullable": False, "primary_key": False},
            {"name": "email", "type": "VARCHAR(255)", "nullable": True, "primary_key": False},
            {"name": "created_at", "type": "TIMESTAMP", "nullable": False, "primary_key": False},
            {"name": "updated_at", "type": "TIMESTAMP", "nullable": True, "primary_key": False},
        ],
        "indexes": [
            {"name": f"idx_{table_name}_email", "columns": ["email"], "unique": True},
        ],
        "row_count_estimate": 15000,
    }

    return (
        f"[PLACEHOLDER] Schema for '{table_name}'.\n"
        f"  Connection: {connection_string or '(default)'}\n"
        f"  Configure a real database for actual schema introspection.\n\n"
        + json.dumps(sample_schema, indent=2)
    )


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULE_TOOLS: dict[str, Tool] = {
    "sql_query": sql_query,
    "nosql_query": nosql_query,
    "describe_table": describe_table,
}


def get_database_tools(names: list[str] | None = None) -> list[Tool]:
    """Get database tools by name. If names is None, return all."""
    if names is None:
        return list(MODULE_TOOLS.values())
    return [MODULE_TOOLS[n] for n in names if n in MODULE_TOOLS]


try:
    from duxx_ai.tools.registry import register_domain

    register_domain("database", MODULE_TOOLS)
except ImportError:
    pass

"""API domain tools for Duxx AI agents.

Provides tools for making REST and GraphQL calls using httpx.
These tools perform real HTTP requests, so they work out of the box
for any reachable endpoint.
"""

from __future__ import annotations

import json

from duxx_ai.core.tool import Tool, tool

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="rest_call",
    description="Make a REST API call to any HTTP endpoint.",
    tags=["api", "http", "rest"],
)
def rest_call(
    url: str,
    method: str = "GET",
    headers: str = "{}",
    body: str = "",
    auth_token: str = "",
) -> str:
    """Execute a REST API request.

    Args:
        url: Full URL to call (e.g. https://api.example.com/v1/users).
        method: HTTP method -- GET, POST, PUT, PATCH, DELETE.
        headers: JSON-encoded dict of additional request headers.
        body: Request body (typically JSON string for POST/PUT/PATCH).
        auth_token: Bearer token for Authorization header (optional).

    Returns:
        Response status, headers summary, and body.
    """
    import httpx

    if not url or not url.strip():
        return "Error: URL is required."

    method = method.upper()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
        return f"Error: unsupported HTTP method '{method}'."

    # Parse headers
    try:
        h = json.loads(headers) if headers else {}
    except json.JSONDecodeError as e:
        return f"Error: invalid headers JSON -- {e}"

    # Add auth header if token provided
    if auth_token:
        h["Authorization"] = f"Bearer {auth_token}"

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.request(method, url, headers=h, content=body if body else None)

        # Format response
        content_type = response.headers.get("content-type", "")
        result_parts = [
            f"Status: {response.status_code} {response.reason_phrase}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(response.content)} bytes",
            "",
        ]

        # Parse response body
        if "json" in content_type:
            try:
                result_parts.append(json.dumps(response.json(), indent=2))
            except Exception:
                result_parts.append(response.text[:10000])
        elif "xml" in content_type or "html" in content_type:
            result_parts.append(response.text[:10000])
        else:
            result_parts.append(response.text[:10000])

        return "\n".join(result_parts)

    except httpx.TimeoutException:
        return f"Error: request to {url} timed out after 30s."
    except httpx.ConnectError as e:
        return f"Error: could not connect to {url} -- {e}"
    except Exception as e:
        return f"Error: REST call failed -- {type(e).__name__}: {e}"


@tool(
    name="graphql_query",
    description="Execute a GraphQL query against an endpoint.",
    tags=["api", "graphql"],
)
def graphql_query(
    endpoint: str,
    query: str,
    variables: str = "{}",
    auth_token: str = "",
) -> str:
    """Execute a GraphQL query or mutation.

    Args:
        endpoint: GraphQL endpoint URL.
        query: The GraphQL query or mutation string.
        variables: JSON-encoded dict of query variables.
        auth_token: Bearer token for Authorization header (optional).

    Returns:
        JSON-formatted GraphQL response (data and/or errors).
    """
    import httpx

    if not endpoint or not endpoint.strip():
        return "Error: GraphQL endpoint URL is required."
    if not query or not query.strip():
        return "Error: GraphQL query string is required."

    # Parse variables
    try:
        vars_dict = json.loads(variables) if variables else {}
    except json.JSONDecodeError as e:
        return f"Error: invalid variables JSON -- {e}"

    # Build request
    gql_headers: dict[str, str] = {"Content-Type": "application/json"}
    if auth_token:
        gql_headers["Authorization"] = f"Bearer {auth_token}"

    payload = json.dumps({"query": query, "variables": vars_dict})

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.post(endpoint, headers=gql_headers, content=payload)

        result_parts = [f"Status: {response.status_code}", ""]

        try:
            body = response.json()
            if "errors" in body:
                result_parts.append("GraphQL Errors:")
                for err in body["errors"]:
                    msg = err.get("message", str(err))
                    result_parts.append(f"  - {msg}")
                result_parts.append("")
            if "data" in body:
                result_parts.append("Data:")
                result_parts.append(json.dumps(body["data"], indent=2))
        except Exception:
            result_parts.append(response.text[:10000])

        return "\n".join(result_parts)

    except httpx.TimeoutException:
        return f"Error: GraphQL request to {endpoint} timed out after 30s."
    except httpx.ConnectError as e:
        return f"Error: could not connect to {endpoint} -- {e}"
    except Exception as e:
        return f"Error: GraphQL call failed -- {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULE_TOOLS: dict[str, Tool] = {
    "rest_call": rest_call,
    "graphql_query": graphql_query,
}


def get_api_tools(names: list[str] | None = None) -> list[Tool]:
    """Get API tools by name. If names is None, return all."""
    if names is None:
        return list(MODULE_TOOLS.values())
    return [MODULE_TOOLS[n] for n in names if n in MODULE_TOOLS]


try:
    from duxx_ai.tools.registry import register_domain

    register_domain("api", MODULE_TOOLS)
except ImportError:
    pass

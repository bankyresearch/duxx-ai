"""Built-in tools for Duxx AI agents."""

from __future__ import annotations

import ast
import json
import math
import operator
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from duxx_ai.core.tool import Tool, tool


@tool(name="python_exec", description="Execute Python code in a sandboxed environment", tags=["code", "compute"])
def python_exec(code: str, timeout: int = 30) -> str:
    """Execute Python code and return stdout/stderr."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir(),
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout}s"
        finally:
            Path(f.name).unlink(missing_ok=True)


@tool(name="bash_exec", description="Execute a bash command", requires_approval=True, tags=["system"])
def bash_exec(command: str, timeout: int = 30) -> str:
    """Execute a bash command and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"


@tool(name="read_file", description="Read the contents of a file", tags=["filesystem"])
def read_file(path: str, max_lines: int = 500) -> str:
    """Read a file and return its contents."""
    try:
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
        if p.stat().st_size > 1_000_000:
            return f"File too large (>{p.stat().st_size} bytes). Use max_lines to limit."
        lines = p.read_text().splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


@tool(name="write_file", description="Write content to a file", requires_approval=True, tags=["filesystem"])
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool(name="list_files", description="List files in a directory", tags=["filesystem"])
def list_files(path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    """List files matching a pattern in a directory."""
    try:
        p = Path(path)
        if not p.is_dir():
            return f"Not a directory: {path}"
        if recursive:
            files = sorted(p.rglob(pattern))
        else:
            files = sorted(p.glob(pattern))
        return "\n".join(str(f) for f in files[:200])
    except Exception as e:
        return f"Error listing files: {e}"


@tool(name="web_request", description="Make an HTTP request to a URL", tags=["web"])
def web_request(url: str, method: str = "GET", headers: str = "{}", body: str = "") -> str:
    """Make an HTTP request and return the response."""
    import httpx

    try:
        h = json.loads(headers) if headers else {}
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            if method.upper() == "GET":
                resp = client.get(url, headers=h)
            elif method.upper() == "POST":
                resp = client.post(url, headers=h, content=body)
            elif method.upper() == "PUT":
                resp = client.put(url, headers=h, content=body)
            elif method.upper() == "DELETE":
                resp = client.delete(url, headers=h)
            else:
                return f"Unsupported method: {method}"

            result = f"Status: {resp.status_code}\n"
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                result += json.dumps(resp.json(), indent=2)
            else:
                result += resp.text[:5000]
            return result
    except Exception as e:
        return f"Request failed: {e}"


@tool(name="json_query", description="Query and transform JSON data using JMESPath-like expressions", tags=["data"])
def json_query(data: str, query: str) -> str:
    """Parse JSON and extract data using dot-notation paths."""
    try:
        obj = json.loads(data)
        parts = query.split(".")
        result: Any = obj
        for part in parts:
            if isinstance(result, dict):
                result = result.get(part)
            elif isinstance(result, list) and part.isdigit():
                result = result[int(part)]
            else:
                return f"Cannot navigate '{part}' in {type(result).__name__}"
        return json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


@tool(name="calculator", description="Evaluate a mathematical expression safely", tags=["math"])
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports basic operations and common functions."""

    # Allowed binary operators
    _BINOPS: dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    # Allowed unary operators
    _UNARYOPS: dict[type, Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    # Allowed comparison operators
    _CMPOPS: dict[type, Any] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    # Allowed math functions (single-arg and multi-arg)
    _ALLOWED_FUNCS: dict[str, Any] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "radians": math.radians,
        "degrees": math.degrees,
        "hypot": math.hypot,
        "pow": pow,
    }
    # Allowed constants
    _ALLOWED_NAMES: dict[str, Any] = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "nan": math.nan,
    }

    def _safe_eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        elif isinstance(node, ast.Name):
            if node.id in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[node.id]
            raise ValueError(f"Unknown name: {node.id}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BINOPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            return _BINOPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARYOPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return _UNARYOPS[op_type](_safe_eval(node.operand))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            fname = node.func.id
            if fname not in _ALLOWED_FUNCS:
                raise ValueError(f"Function not allowed: {fname}")
            args = [_safe_eval(arg) for arg in node.args]
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
            return _ALLOWED_FUNCS[fname](*args)
        elif isinstance(node, ast.Compare):
            left = _safe_eval(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=False):
                op_type = type(op)
                if op_type not in _CMPOPS:
                    raise ValueError(f"Unsupported comparison: {op_type.__name__}")
                right = _safe_eval(comparator)
                if not _CMPOPS[op_type](left, right):
                    return False
                left = right
            return True
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Registry of all built-in tools
BUILTIN_TOOLS: dict[str, Tool] = {
    "python_exec": python_exec,
    "bash_exec": bash_exec,
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "web_request": web_request,
    "json_query": json_query,
    "calculator": calculator,
}


def get_builtin_tools(names: list[str] | None = None) -> list[Tool]:
    """Get built-in tools by name. If names is None, return all."""
    if names is None:
        return list(BUILTIN_TOOLS.values())
    return [BUILTIN_TOOLS[n] for n in names if n in BUILTIN_TOOLS]

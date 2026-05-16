"""MCP (Model Context Protocol) integration for Duxx AI.

Provides bidirectional MCP support:
1. MCPClient — Connect to MCP servers and use their tools/resources/prompts
2. MCPToolkit — Manage multiple MCP server connections concurrently
3. MCPServer — Expose Duxx AI tools as an MCP server

Supports all 3 MCP transport protocols:
- stdio (local subprocess servers)
- HTTP/SSE (remote servers)
- Streamable HTTP (modern remote servers)

Requires: pip install mcp

Usage:
    # Connect to an MCP server and load its tools
    async with MCPClient("stdio", command="python", args=["server.py"]) as client:
        tools = await client.load_tools()
        # tools are native Duxx AI Tool objects — use them like any other tool

    # Multi-server management
    toolkit = MCPToolkit({
        "math": {"transport": "stdio", "command": "python", "args": ["math_server.py"]},
        "weather": {"transport": "http", "url": "http://localhost:8000/mcp"},
    })
    async with toolkit:
        all_tools = await toolkit.get_tools()

    # Expose Duxx AI tools as MCP server
    server = MCPServer("my-agent", tools=[calculator, web_search])
    server.run(transport="stdio")
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any

from duxx_ai.core.tool import Tool, ToolParameter

logger = logging.getLogger(__name__)


# ── MCP Types ──

@dataclass
class MCPToolDefinition:
    """Raw MCP tool definition from a server."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPResource:
    """MCP resource (read-only data from server)."""
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = "text/plain"
    content: str = ""


@dataclass
class MCPPrompt:
    """MCP prompt template from server."""
    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MCPServerInfo:
    """Information about a connected MCP server."""
    name: str
    version: str = ""
    protocol_version: str = "2024-11-05"
    capabilities: dict[str, Any] = field(default_factory=dict)
    tools: list[MCPToolDefinition] = field(default_factory=list)
    resources: list[MCPResource] = field(default_factory=list)
    prompts: list[MCPPrompt] = field(default_factory=list)
    status: str = "disconnected"  # disconnected, connecting, connected, error


# ── MCP Client ──

class MCPClient:
    """Connect to a single MCP server and interact with its tools/resources/prompts.

    Supports stdio, HTTP, and SSE transports.

    Usage:
        # stdio transport (local server)
        async with MCPClient("stdio", command="python", args=["server.py"]) as client:
            tools = await client.load_tools()
            result = await client.call_tool("add", {"a": 1, "b": 2})

        # HTTP transport (remote server)
        async with MCPClient("http", url="http://localhost:8000/mcp") as client:
            tools = await client.load_tools()
    """

    def __init__(
        self,
        transport: str = "stdio",
        *,
        command: str = "",
        args: list[str] | None = None,
        url: str = "",
        headers: dict[str, str] | None = None,
        server_name: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.transport = transport
        self.command = command
        self.args = args or []
        self.url = url
        self.headers = headers or {}
        self.server_name = server_name or f"mcp-{uuid.uuid4().hex[:6]}"
        self.timeout = timeout
        self._session: Any = None
        self._process: subprocess.Popen | None = None
        self._info = MCPServerInfo(name=self.server_name)
        self._connected = False

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()

    async def connect(self) -> MCPServerInfo:
        """Establish connection to the MCP server."""
        self._info.status = "connecting"
        try:
            if self.transport == "stdio":
                await self._connect_stdio()
            elif self.transport in ("http", "sse", "streamable_http"):
                await self._connect_http()
            else:
                raise ValueError(f"Unknown transport: {self.transport}. Use 'stdio', 'http', or 'sse'.")

            self._connected = True
            self._info.status = "connected"
            logger.info(f"Connected to MCP server '{self.server_name}' via {self.transport}")
            return self._info

        except Exception as e:
            self._info.status = "error"
            logger.error(f"Failed to connect to MCP server '{self.server_name}': {e}")
            raise

    async def _connect_stdio(self) -> None:
        """Connect via stdio (subprocess)."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=self.command, args=self.args)
            read, write = await asyncio.wait_for(
                stdio_client(params).__aenter__(), timeout=self.timeout
            )
            self._session = ClientSession(read, write)
            await self._session.initialize()
            await self._discover_capabilities()

        except ImportError:
            # Fallback: direct subprocess communication
            logger.warning("MCP SDK not installed. Using basic stdio fallback.")
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            # Send initialize request
            init_req = {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"roots": {}, "sampling": {}},
                    "clientInfo": {"name": "duxx-ai", "version": "0.4.0"},
                },
            }
            self._send_jsonrpc(init_req)
            resp = self._recv_jsonrpc()
            if resp and "result" in resp:
                self._info.protocol_version = resp["result"].get("protocolVersion", "")
                self._info.capabilities = resp["result"].get("capabilities", {})
                self._info.version = resp["result"].get("serverInfo", {}).get("version", "")
            # Send initialized notification
            self._send_jsonrpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
            await self._discover_capabilities_raw()

    async def _connect_http(self) -> None:
        """Connect via HTTP/SSE."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            read, write = await asyncio.wait_for(
                sse_client(self.url, headers=self.headers).__aenter__(), timeout=self.timeout
            )
            self._session = ClientSession(read, write)
            await self._session.initialize()
            await self._discover_capabilities()

        except ImportError:
            # Fallback: direct HTTP
            logger.warning("MCP SDK not installed. Using basic HTTP fallback.")
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                # Initialize
                resp = await http.post(
                    self.url,
                    headers={**self.headers, "Content-Type": "application/json"},
                    json={
                        "jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "duxx-ai", "version": "0.4.0"},
                        },
                    },
                )
                data = resp.json()
                if "result" in data:
                    self._info.capabilities = data["result"].get("capabilities", {})
                await self._discover_capabilities_http()

    async def _discover_capabilities(self) -> None:
        """Discover server capabilities using MCP SDK session."""
        if not self._session:
            return

        # List tools
        if self._info.capabilities.get("tools"):
            result = await self._session.list_tools()
            self._info.tools = [
                MCPToolDefinition(
                    name=t.name, description=t.description or "",
                    input_schema=t.inputSchema if hasattr(t, 'inputSchema') else {},
                    server_name=self.server_name,
                )
                for t in result.tools
            ]

        # List resources
        if self._info.capabilities.get("resources"):
            result = await self._session.list_resources()
            self._info.resources = [
                MCPResource(uri=r.uri, name=r.name or "", description=r.description or "")
                for r in result.resources
            ]

        # List prompts
        if self._info.capabilities.get("prompts"):
            result = await self._session.list_prompts()
            self._info.prompts = [
                MCPPrompt(name=p.name, description=p.description or "", arguments=p.arguments or [])
                for p in result.prompts
            ]

    async def _discover_capabilities_raw(self) -> None:
        """Discover capabilities without MCP SDK (raw JSON-RPC)."""
        # List tools
        if self._info.capabilities.get("tools"):
            self._send_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            resp = self._recv_jsonrpc()
            if resp and "result" in resp:
                for t in resp["result"].get("tools", []):
                    self._info.tools.append(MCPToolDefinition(
                        name=t["name"], description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}), server_name=self.server_name,
                    ))

    async def _discover_capabilities_http(self) -> None:
        """Discover capabilities via HTTP without MCP SDK."""
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            if self._info.capabilities.get("tools"):
                resp = await http.post(
                    self.url, headers={**self.headers, "Content-Type": "application/json"},
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                )
                data = resp.json()
                for t in data.get("result", {}).get("tools", []):
                    self._info.tools.append(MCPToolDefinition(
                        name=t["name"], description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}), server_name=self.server_name,
                    ))

    def _send_jsonrpc(self, msg: dict) -> None:
        """Send JSON-RPC message via stdio."""
        if self._process and self._process.stdin:
            self._process.stdin.write(json.dumps(msg) + "\n")
            self._process.stdin.flush()

    def _recv_jsonrpc(self) -> dict | None:
        """Receive JSON-RPC response via stdio."""
        if self._process and self._process.stdout:
            line = self._process.stdout.readline().strip()
            if line:
                return json.loads(line)
        return None

    async def disconnect(self) -> None:
        """Close the MCP server connection."""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._process:
            self._process.terminate()
            self._process = None
        self._connected = False
        self._info.status = "disconnected"
        logger.info(f"Disconnected from MCP server '{self.server_name}'")

    # ── Tool Operations ──

    async def load_tools(self) -> list[Tool]:
        """Convert MCP tools to native Duxx AI Tool objects.

        Returns:
            List of Tool objects that can be used with any Duxx AI Agent.
        """
        if not self._info.tools:
            if self._connected:
                await self._discover_capabilities()
            if not self._info.tools:
                return []

        tools = []
        for mcp_tool in self._info.tools:
            tool = _mcp_to_duxx_tool(mcp_tool, self)
            tools.append(tool)

        logger.info(f"Loaded {len(tools)} tools from MCP server '{self.server_name}'")
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call an MCP tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if self._session:
            result = await self._session.call_tool(name, arguments or {})
            # Extract content from result
            if hasattr(result, 'content') and result.content:
                texts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(texts) if texts else str(result.content)
            return str(result)
        elif self._process:
            # Raw JSON-RPC call
            call_id = uuid.uuid4().int % 100000
            self._send_jsonrpc({
                "jsonrpc": "2.0", "id": call_id, "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            })
            resp = self._recv_jsonrpc()
            if resp and "result" in resp:
                content = resp["result"].get("content", [])
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts) if texts else str(content)
            elif resp and "error" in resp:
                raise RuntimeError(f"MCP tool error: {resp['error']}")
        else:
            raise RuntimeError("Not connected to MCP server")

    # ── Resource Operations ──

    async def read_resource(self, uri: str) -> str:
        """Read an MCP resource by URI."""
        if self._session:
            result = await self._session.read_resource(uri)
            if hasattr(result, 'contents') and result.contents:
                texts = [c.text for c in result.contents if hasattr(c, 'text')]
                return "\n".join(texts) if texts else str(result.contents)
            return str(result)
        raise RuntimeError("Resource reading requires MCP SDK session")

    async def list_resources(self) -> list[MCPResource]:
        """List available resources."""
        return self._info.resources

    # ── Prompt Operations ──

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Get a rendered MCP prompt."""
        if self._session:
            result = await self._session.get_prompt(name, arguments or {})
            if hasattr(result, 'messages'):
                texts = []
                for msg in result.messages:
                    if hasattr(msg, 'content') and hasattr(msg.content, 'text'):
                        texts.append(msg.content.text)
                return "\n".join(texts)
            return str(result)
        raise RuntimeError("Prompt fetching requires MCP SDK session")

    async def list_prompts(self) -> list[MCPPrompt]:
        """List available prompts."""
        return self._info.prompts

    @property
    def info(self) -> MCPServerInfo:
        return self._info

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── MCP Tool Conversion ──

def _mcp_to_duxx_tool(mcp_tool: MCPToolDefinition, client: MCPClient) -> Tool:
    """Convert an MCP tool definition to a native Duxx AI Tool."""
    # Parse input schema into ToolParameter list
    parameters = []
    schema = mcp_tool.input_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "string")
        parameters.append(ToolParameter(
            name=param_name,
            type=param_type,
            description=param_info.get("description", ""),
            required=param_name in required,
            default=param_info.get("default"),
        ))

    # Create the tool
    tool = Tool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"MCP tool from {mcp_tool.server_name}",
        parameters=parameters,
        tags=["mcp", mcp_tool.server_name],
    )

    # Bind async execution function
    async def _execute(call: Any) -> Any:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return await client.call_tool(mcp_tool.name, args)

    tool.bind(_execute)
    return tool


# ── MCPToolkit (Multi-Server) ──

class MCPToolkit:
    """Manage multiple MCP server connections concurrently.

    Usage:
        toolkit = MCPToolkit({
            "math": {"transport": "stdio", "command": "python", "args": ["math.py"]},
            "weather": {"transport": "http", "url": "http://localhost:8000/mcp"},
        })
        async with toolkit:
            all_tools = await toolkit.get_tools()
            math_tools = await toolkit.get_tools(server="math")

            # Use with Agent
            agent = Agent(tools=all_tools, ...)
    """

    def __init__(self, connections: dict[str, dict[str, Any]]) -> None:
        self.connections = connections
        self._clients: dict[str, MCPClient] = {}

    async def __aenter__(self) -> MCPToolkit:
        await self.connect_all()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect_all()

    async def connect_all(self) -> dict[str, MCPServerInfo]:
        """Connect to all configured MCP servers."""
        results = {}
        tasks = []
        for name, config in self.connections.items():
            client = MCPClient(
                transport=config.get("transport", "stdio"),
                command=config.get("command", ""),
                args=config.get("args", []),
                url=config.get("url", ""),
                headers=config.get("headers", {}),
                server_name=name,
            )
            self._clients[name] = client
            tasks.append(client.connect())

        infos = await asyncio.gather(*tasks, return_exceptions=True)
        for name, info in zip(self.connections, infos, strict=False):
            if isinstance(info, Exception):
                logger.error(f"Failed to connect to '{name}': {info}")
                results[name] = MCPServerInfo(name=name, status="error")
            else:
                results[name] = info
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        tasks = [c.disconnect() for c in self._clients.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()

    async def get_tools(self, server: str | None = None) -> list[Tool]:
        """Get tools from all servers or a specific server.

        Args:
            server: If specified, only return tools from this server

        Returns:
            List of native Duxx AI Tool objects
        """
        if server:
            client = self._clients.get(server)
            if not client:
                raise ValueError(f"Server '{server}' not found")
            return await client.load_tools()

        all_tools = []
        for client in self._clients.values():
            try:
                tools = await client.load_tools()
                all_tools.extend(tools)
            except Exception as e:
                logger.error(f"Failed to load tools from '{client.server_name}': {e}")
        return all_tools

    async def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool on a specific server."""
        client = self._clients.get(server)
        if not client:
            raise ValueError(f"Server '{server}' not found")
        return await client.call_tool(tool_name, arguments)

    def get_server_info(self, server: str | None = None) -> dict[str, MCPServerInfo] | MCPServerInfo:
        """Get server connection info."""
        if server:
            client = self._clients.get(server)
            return client.info if client else MCPServerInfo(name=server, status="not_found")
        return {name: c.info for name, c in self._clients.items()}

    @property
    def connected_servers(self) -> list[str]:
        return [name for name, c in self._clients.items() if c.is_connected]


# ── MCP Server Builder ──

class MCPServer:
    """Expose Duxx AI tools as an MCP server.

    This allows any MCP client (Claude Desktop, Cursor, etc.) to use
    Duxx AI tools natively.

    Usage:
        from duxx_ai.tools.builtin import get_builtin_tools
        from duxx_ai.tools.mcp import MCPServer

        tools = get_builtin_tools(["calculator", "web_request"])
        server = MCPServer("duxx-agent", tools=tools)
        server.run(transport="stdio")  # or transport="sse", port=8000
    """

    def __init__(self, name: str = "duxx-ai", tools: list[Tool] | None = None, version: str = "0.4.0") -> None:
        self.name = name
        self.tools = tools or []
        self.version = version

    def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start the MCP server.

        Args:
            transport: "stdio", "sse", or "streamable_http"
            host: HTTP host (for sse/http transports)
            port: HTTP port (for sse/http transports)
        """
        try:
            from mcp.server.fastmcp import FastMCP
            mcp = FastMCP(self.name)

            # Register each Duxx AI tool as an MCP tool
            for tool in self.tools:
                self._register_tool(mcp, tool)

            logger.info(f"Starting MCP server '{self.name}' with {len(self.tools)} tools via {transport}")

            if transport == "stdio":
                mcp.run(transport="stdio")
            elif transport in ("sse", "streamable_http"):
                mcp.run(transport=transport, host=host, port=port)
            else:
                raise ValueError(f"Unknown transport: {transport}")

        except ImportError:
            # Fallback: basic JSON-RPC server
            logger.warning("MCP SDK not installed. Using basic JSON-RPC server.")
            if transport == "stdio":
                self._run_stdio_basic()
            else:
                raise ImportError("MCP SDK required for HTTP transport: pip install mcp")

    def _register_tool(self, mcp: Any, tool: Tool) -> None:
        """Register a single Duxx AI tool with FastMCP."""
        _schema = tool.to_schema()  # reserved for future schema-aware registration

        @mcp.tool(name=tool.name, description=tool.description)
        async def _handler(**kwargs: Any) -> str:
            try:
                result = await tool.execute({"name": tool.name, "arguments": kwargs})
                return str(result.result if hasattr(result, 'result') else result)
            except Exception as e:
                return f"Error: {e}"

    def _run_stdio_basic(self) -> None:
        """Basic stdio MCP server without SDK."""
        import sys

        def handle_request(req: dict) -> dict:
            method = req.get("method", "")
            req_id = req.get("id")

            if method == "initialize":
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": self.name, "version": self.version},
                    },
                }
            elif method == "tools/list":
                tools_list = []
                for tool in self.tools:
                    schema = tool.to_schema()
                    tools_list.append({
                        "name": schema["name"],
                        "description": schema.get("description", ""),
                        "inputSchema": schema.get("parameters", {"type": "object", "properties": {}}),
                    })
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_list}}
            elif method == "tools/call":
                params = req.get("params", {})
                name = params.get("name", "")
                args = params.get("arguments", {})
                for tool in self.tools:
                    if tool.name == name:
                        try:
                            import asyncio
                            result = asyncio.run(tool.execute({"name": name, "arguments": args}))
                            text = str(result.result if hasattr(result, 'result') else result)
                            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": text}]}}
                        except Exception as e:
                            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}}
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Tool '{name}' not found"}}
            elif method == "notifications/initialized":
                return None  # Notification, no response
            else:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method '{method}' not found"}}

        logger.info(f"MCP Server '{self.name}' running on stdio with {len(self.tools)} tools")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = handle_request(req)
                if resp:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError:
                pass
            except Exception as e:
                err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()

    def to_config(self, transport: str = "stdio") -> dict[str, Any]:
        """Generate MCP client config for connecting to this server.

        Returns a dict suitable for MCPToolkit connections config.
        """
        if transport == "stdio":
            return {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "duxx_ai.tools.mcp_server", "--name", self.name],
            }
        return {
            "transport": transport,
            "url": "http://127.0.0.1:8000/mcp",
        }

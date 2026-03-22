"""Duxx AI CLI — command-line interface for managing agents and workflows."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="duxx_ai")
def cli() -> None:
    """Duxx AI — Enterprise Agentic SDK"""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--input", "-i", "user_input", prompt="Enter your message", help="User input")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def run(config_path: str, user_input: str, verbose: bool) -> None:
    """Run an agent from a YAML/JSON config file."""
    from duxx_ai.core.agent import Agent, AgentConfig
    from duxx_ai.core.llm import LLMConfig
    from duxx_ai.tools.builtin import get_builtin_tools

    config_file = Path(config_path)
    if config_file.suffix in (".yaml", ".yml"):
        import yaml
        with open(config_file) as f:
            raw = yaml.safe_load(f)
    else:
        with open(config_file) as f:
            raw = json.load(f)

    agent_config = AgentConfig(**raw.get("agent", raw))
    tools = get_builtin_tools(raw.get("tools"))

    agent = Agent(config=agent_config, tools=tools)

    async def _run() -> str:
        return await agent.run(user_input)

    result = asyncio.run(_run())
    console.print(Panel(result, title=f"[bold green]{agent.name}[/bold green]", border_style="green"))


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8080, help="Port to bind to")
def studio(host: str, port: int) -> None:
    """Launch Duxx AI Studio web UI."""
    console.print(f"[bold]Starting Duxx AI Studio on {host}:{port}[/bold]")
    try:
        import uvicorn
        from duxx_ai.studio.app import create_app
        app = create_app()
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        console.print("[red]Studio requires: pip install duxx_ai[studio][/red]")


@cli.command()
def init() -> None:
    """Initialize a new Duxx AI project."""
    project_dir = Path(".")
    config = {
        "agent": {
            "name": "my-agent",
            "description": "A Duxx AI agent",
            "system_prompt": "You are a helpful AI assistant.",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.7,
            },
        },
        "tools": ["python_exec", "read_file", "write_file", "calculator"],
        "guardrails": {
            "pii_filter": True,
            "prompt_injection": True,
            "token_budget": 100000,
        },
    }

    config_path = project_dir / "duxx_ai.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"[green]Created {config_path}[/green]")
    console.print("Run [bold]duxx_ai run duxx_ai.json[/bold] to start your agent")


@cli.group()
def finetune() -> None:
    """Fine-tuning commands."""
    pass


@finetune.command("prepare")
@click.argument("traces_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="dataset.jsonl", help="Output dataset path")
@click.option("--format", "fmt", default="chat", type=click.Choice(["chat", "instruction"]))
def finetune_prepare(traces_path: str, output: str, fmt: str) -> None:
    """Prepare a training dataset from agent traces."""
    from duxx_ai.finetune.pipeline import TraceToDataset
    count = TraceToDataset.from_traces(traces_path, output, format=fmt)
    console.print(f"[green]Generated {count} training samples -> {output}[/green]")


@finetune.command("train")
@click.option("--model", default="unsloth/Qwen2.5-7B", help="Base model")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Training dataset")
@click.option("--output", "-o", default="./duxx_ai-finetuned", help="Output directory")
@click.option("--epochs", default=3, help="Number of epochs")
@click.option("--method", default="lora", type=click.Choice(["lora", "qlora", "full"]))
def finetune_train(model: str, dataset: str, output: str, epochs: int, method: str) -> None:
    """Fine-tune a model on your dataset."""
    from duxx_ai.finetune.pipeline import FineTunePipeline, TrainingConfig

    config = TrainingConfig(
        base_model=model,
        output_dir=output,
        method=method,
        epochs=epochs,
    )
    pipeline = FineTunePipeline(training_config=config)

    with console.status("[bold green]Training...[/bold green]"):
        result = pipeline.train(dataset)

    console.print(f"[green]Training complete![/green]")
    console.print(f"  Model: {result.model_path}")
    console.print(f"  Loss: {result.final_loss:.4f}")
    console.print(f"  Time: {result.total_time_seconds:.1f}s")


@cli.group()
def templates() -> None:
    """Manage enterprise agent templates."""
    pass


@templates.command("list")
@click.option("--category", "-c", default=None, help="Filter by category")
def templates_list(category: str | None) -> None:
    """List all available agent templates."""
    from duxx_ai.templates import TEMPLATES

    table = Table(title="Duxx AI Enterprise Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description", style="white")
    table.add_column("Tools", style="dim")

    for name, cls in TEMPLATES.items():
        tpl_info = cls.info()
        if category and tpl_info["category"].lower() != category.lower():
            continue
        tools_str = ", ".join(tpl_info["default_tools"][:3])
        if len(tpl_info["default_tools"]) > 3:
            tools_str += "..."
        table.add_row(name, tpl_info["category"], tpl_info["description"][:60], tools_str)

    console.print(table)


@templates.command("info")
@click.argument("template_name")
def templates_info(template_name: str) -> None:
    """Show detailed information about a template."""
    from duxx_ai.templates import TEMPLATES

    cls = TEMPLATES.get(template_name)
    if cls is None:
        console.print(f"[red]Template '{template_name}' not found.[/red]")
        console.print(f"Available: {', '.join(TEMPLATES.keys())}")
        return

    tpl_info = cls.info()
    console.print(Panel(
        f"[bold]{tpl_info['name']}[/bold]\n\n"
        f"Category: {tpl_info['category']}\n"
        f"Description: {tpl_info['description']}\n\n"
        f"Default Tools: {', '.join(tpl_info['default_tools'])}",
        title=f"Template: {template_name}",
        border_style="cyan",
    ))


@templates.command("run")
@click.argument("template_name")
@click.option("--input", "-i", "user_input", prompt="Enter your message", help="User input")
@click.option("--model", "-m", default="gpt-4o", help="Model to use")
@click.option("--provider", "-p", default="openai", help="LLM provider")
def templates_run(template_name: str, user_input: str, model: str, provider: str) -> None:
    """Run a template agent with a message."""
    from duxx_ai.templates import TEMPLATES
    from duxx_ai.core.llm import LLMConfig

    cls = TEMPLATES.get(template_name)
    if cls is None:
        console.print(f"[red]Template '{template_name}' not found.[/red]")
        return

    llm_config = LLMConfig(provider=provider, model=model)
    agent = cls.create(llm_config=llm_config)

    async def _run() -> str:
        return await agent.run(user_input)

    result = asyncio.run(_run())
    console.print(Panel(result, title=f"[bold green]{template_name}[/bold green]", border_style="green"))


@cli.group()
def tools() -> None:
    """Manage tools and tool domains."""
    pass


@tools.command("list")
@click.option("--domain", "-d", default=None, help="Filter by domain")
def tools_list(domain: str | None) -> None:
    """List all available tools."""
    from duxx_ai.tools.builtin import BUILTIN_TOOLS

    table = Table(title="Duxx AI Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Domain", style="green")
    table.add_column("Description", style="white")
    table.add_column("Approval", style="yellow")

    if not domain or domain == "builtin":
        for name, t in BUILTIN_TOOLS.items():
            table.add_row(name, "builtin", t.description[:50], "Yes" if t.requires_approval else "No")

    try:
        from duxx_ai.tools.registry import DOMAIN_TOOLS
        for dom, dom_tools in DOMAIN_TOOLS.items():
            if domain and dom != domain:
                continue
            for name, t in dom_tools.items():
                table.add_row(name, dom, t.description[:50], "Yes" if t.requires_approval else "No")
    except ImportError:
        pass

    console.print(table)


@cli.command()
def info() -> None:
    """Show Duxx AI system information."""
    import duxx_ai

    table = Table(title="Duxx AI System Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", duxx_ai.__version__)
    table.add_row("Python", sys.version.split()[0])

    # Check optional dependencies
    deps = {
        "torch": "Fine-tuning",
        "transformers": "Fine-tuning",
        "fastapi": "Studio",
        "opentelemetry": "Tracing",
    }
    for pkg, feature in deps.items():
        try:
            __import__(pkg)
            table.add_row(f"{feature} ({pkg})", "installed")
        except ImportError:
            table.add_row(f"{feature} ({pkg})", "[dim]not installed[/dim]")

    console.print(table)


if __name__ == "__main__":
    cli()

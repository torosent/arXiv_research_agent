"""Command-line interface for the arXiv Research Agent.

This script provides a simple CLI to start research and view results.
"""

import argparse
import asyncio
import json
import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from azure.identity import DefaultAzureCredential
from durabletask import client as durable_client
from durabletask.azuremanaged.client import DurableTaskSchedulerClient

console = Console()


def get_client() -> DurableTaskSchedulerClient:
    """Create DurableTask client."""
    taskhub_name = os.getenv("TASKHUB", "default")
    endpoint = os.getenv("ENDPOINT", "http://localhost:8080")
    
    credential = None
    if endpoint != "http://localhost:8080":
        credential = DefaultAzureCredential()
    
    return DurableTaskSchedulerClient(
        host_address=endpoint,
        secure_channel=endpoint != "http://localhost:8080",
        taskhub=taskhub_name,
        token_credential=credential
    )


def start_research(topic: str, max_iterations: int = 3) -> str:
    """Start a new research agent."""
    client = get_client()
    
    instance_id = client.schedule_new_orchestration(
        "agentic_research_orchestrator",
        input={
            "topic": topic,
            "max_iterations": max_iterations
        }
    )
    
    return instance_id


def wait_for_result(instance_id: str, timeout: int = 300):
    """Wait for research to complete and return result."""
    client = get_client()
    
    state = client.wait_for_orchestration_completion(
        instance_id,
        timeout=timeout
    )
    
    if state is None:
        return None
    
    if state.runtime_status == durable_client.OrchestrationStatus.COMPLETED:
        if state.serialized_output:
            return json.loads(state.serialized_output)
        return None
    
    return None


def research_command(args):
    """Handle research command."""
    topic = args.topic
    max_iterations = args.iterations
    
    console.print(f"\n[bold blue]🔬 Starting research on: {topic}[/bold blue]\n")
    
    # Start the research
    instance_id = start_research(topic, max_iterations)
    console.print(f"[dim]Instance ID: {instance_id}[/dim]\n")
    
    if args.nowait:
        console.print("[green]✅ Research started in background[/green]")
        console.print(f"[dim]Check status: python -m arxiv_research_agent.cli status {instance_id}[/dim]")
        return
    
    # Wait for completion with progress
    console.print("[yellow]⏳ Researching arXiv papers... (this may take a few minutes)[/yellow]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Researching...", total=None)
        
        result = wait_for_result(instance_id, timeout=args.timeout)
    
    if result is None:
        console.print("[red]❌ Research timed out or failed[/red]")
        return
    
    # Display results
    console.print(f"\n[bold green]✅ Research Complete![/bold green]")
    console.print(f"[dim]Topic: {result.get('topic', 'Unknown')}[/dim]")
    console.print(f"[dim]Iterations: {result.get('iterations', 0)}[/dim]")
    console.print(f"[dim]Findings: {result.get('findings_count', 0)}[/dim]\n")
    
    report = result.get("report", "No report generated")
    console.print(Panel(
        Markdown(report),
        title="📊 Research Report",
        border_style="blue",
        padding=(1, 2)
    ))


def status_command(args):
    """Handle status command."""
    client = get_client()
    
    state = client.get_orchestration_state(args.instance_id)
    
    if state is None:
        console.print("[red]❌ Instance not found[/red]")
        return
    
    status_colors = {
        durable_client.OrchestrationStatus.PENDING: "yellow",
        durable_client.OrchestrationStatus.RUNNING: "blue",
        durable_client.OrchestrationStatus.COMPLETED: "green",
        durable_client.OrchestrationStatus.FAILED: "red",
    }
    
    color = status_colors.get(state.runtime_status, "white")
    console.print(f"[{color}]Status: {state.runtime_status.name}[/{color}]")
    
    if state.runtime_status == durable_client.OrchestrationStatus.COMPLETED:
        if state.serialized_output:
            result = json.loads(state.serialized_output)
            console.print(f"\n[bold]Report:[/bold]")
            console.print(Panel(
                Markdown(result.get("report", "No report")),
                border_style="blue"
            ))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="arXiv Research Agent CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Research command
    research_parser = subparsers.add_parser(
        "research",
        help="Start a new research agent"
    )
    research_parser.add_argument(
        "topic",
        help="Topic to research"
    )
    research_parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=3,
        help="Maximum research iterations (default: 3)"
    )
    research_parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)"
    )
    research_parser.add_argument(
        "--nowait",
        action="store_true",
        help="Don't wait for completion"
    )
    research_parser.set_defaults(func=research_command)
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check status of a research agent"
    )
    status_parser.add_argument(
        "instance_id",
        help="Orchestration instance ID"
    )
    status_parser.set_defaults(func=status_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()

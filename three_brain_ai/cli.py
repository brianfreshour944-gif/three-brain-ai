"""CLI entry point for three-brain AI system."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from orchestrator import (
    ThreeBrainOrchestrator,
    create_orchestrator,
    get_task_store,
    get_memory_manager,
)
from orchestrator.safety import run_safety_check

# FastAPI for serve command
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

app = typer.Typer(help="Three-Brain AI Orchestrator")
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description"),
    project_root: Optional[Path] = typer.Option(None, "--project", "-p", help="Project root directory"),
    notes: str = typer.Option("", "--notes", "-n", help="Additional context/notes"),
    auto_approve: bool = typer.Option(False, "--auto-approve", "-a", help="Skip human approval"),
    no_red_team: bool = typer.Option(False, "--no-red-team", help="Disable red team review"),
    files: Optional[str] = typer.Option(None, "--files", "-f", help="Comma-separated relevant files"),
):
    """Run a task through the three-brain pipeline."""
    root = project_root or Path.cwd()

    console.print(Panel.fit(
        f"[bold]Three-Brain Orchestrator[/bold]\n"
        f"Project: {root}\n"
        f"Task: {task}",
        title="Starting",
        border_style="blue",
    ))

    # Parse files
    relevant_files = []
    if files:
        relevant_files = [Path(f.strip()) for f in files.split(",")]

    orchestrator = create_orchestrator(
        project_root=root,
        enable_red_team=not no_red_team,
    )

    # Check LLM health
    console.print("\n[cyan]Checking LLM endpoints...[/cyan]")
    health = asyncio.run(orchestrator.check_llm_health())

    health_table = Table(title="LLM Health")
    health_table.add_column("Model", style="cyan")
    health_table.add_column("Status", style="green")
    for model, status in health.items():
        health_table.add_row(model, "✓ Healthy" if status else "✗ Unreachable")
    console.print(health_table)

    if not all(health.values()):
        console.print("[red]Some LLM endpoints are unreachable. Continue anyway?[/red]")
        if not Confirm.ask("Continue?"):
            raise typer.Exit(1)

    # Run task
    try:
        result = asyncio.run(orchestrator.run_task(
            task_description=task,
            user_notes=notes,
            relevant_files=relevant_files,
            auto_approve=auto_approve,
        ))

        # Display results
        console.print("\n" + "=" * 60)
        console.print(Panel(
            Markdown(result.final_plan),
            title=f"Final Plan (Task: {result.task.id})",
            border_style="green",
            expand=False,
        ))

        if result.requires_approval:
            console.print("\n[yellow]Human approval required.[/yellow]")
            console.print(f"Task ID: {result.task.id}")
            console.print("Use 'three-brain approve <task_id>' to approve.")
            console.print("Use 'three-brain reject <task_id>' to reject.")
            console.print("Use 'three-brain revise <task_id>' to request revisions.")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def approve(
    task_id: str = typer.Argument(..., help="Task ID to approve"),
    notes: str = typer.Option("", "--notes", "-n", help="Approval notes"),
):
    """Approve a task for implementation."""
    orchestrator = create_orchestrator()
    task = orchestrator.approve_task(task_id, notes)
    console.print(f"[green]✓ Task {task_id} approved[/green]")
    console.print(f"Status: {task.status}")
    if notes:
        console.print(f"Notes: {notes}")


@app.command()
def reject(
    task_id: str = typer.Argument(..., help="Task ID to reject"),
    notes: str = typer.Option(..., "--notes", "-n", help="Rejection reason"),
):
    """Reject a task."""
    orchestrator = create_orchestrator()
    task = orchestrator.reject_task(task_id, notes)
    console.print(f"[red]✗ Task {task_id} rejected[/red]")
    console.print(f"Reason: {notes}")


@app.command()
def revise(
    task_id: str = typer.Argument(..., help="Task ID to revise"),
    notes: str = typer.Option(..., "--notes", "-n", help="Revision notes"),
):
    """Request revisions to a task."""
    orchestrator = create_orchestrator()
    task = orchestrator.request_revisions(task_id, notes)
    console.print(f"[yellow]↻ Task {task_id} revision requested[/yellow]")
    console.print(f"Notes: {notes}")


@app.command()
def status(
    task_id: Optional[str] = typer.Argument(None, help="Task ID to check"),
    all_tasks: bool = typer.Option(False, "--all", help="Show all tasks"),
):
    """Check task status."""
    orchestrator = create_orchestrator()

    if task_id:
        task = orchestrator.get_task_status(task_id)
        if not task:
            console.print(f"[red]Task {task_id} not found[/red]")
            raise typer.Exit(1)

        console.print(Panel(
            f"ID: {task.id}\n"
            f"Description: {task.description}\n"
            f"Status: {task.status}\n"
            f"Created: {task.created_at}\n"
            f"Updated: {task.updated_at}\n"
            f"Approval: {task.approval_status or 'N/A'}\n"
            f"Error: {task.error or 'None'}",
            title=f"Task {task_id}",
            border_style="blue",
        ))

        if task.final_plan:
            console.print("\n[bold]Final Plan:[/bold]")
            console.print(Markdown(task.final_plan))

    elif all_tasks:
        tasks = orchestrator.list_tasks()
        if not tasks:
            console.print("No tasks found.")
            return

        table = Table(title="All Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Created", style="dim")
        for task in tasks:
            table.add_row(
                task.id,
                task.description[:60] + "..." if len(task.description) > 60 else task.description,
                task.status,
                task.created_at[:19],
            )
        console.print(table)

    else:
        # Show latest
        task = orchestrator.get_task_status(
            orchestrator.task_store.get_latest().id
        ) if orchestrator.task_store.get_latest() else None

        if task:
            console.print(f"Latest task: {task.id} - {task.status}")
        else:
            console.print("No tasks found.")


@app.command()
def memory(
    show: bool = typer.Option(False, "--show", help="Show current memory"),
    clear: bool = typer.Option(False, "--clear", help="Clear all memory"),
    add_pref: Optional[str] = typer.Option(None, "--add-pref", help="Add preference (key=value)"),
    add_lesson: Optional[str] = typer.Option(None, "--add-lesson", help="Add lesson learned"),
):
    """Manage project memory."""
    memory = get_memory_manager()

    if clear:
        if Confirm.ask("Clear all project memory?"):
            memory.clear()
            console.print("[green]Memory cleared[/green]")
        return

    if add_pref:
        key, value = add_pref.split("=", 1) if "=" in add_pref else (add_pref, "")
        memory.add_preference(key.strip(), value.strip())
        console.print(f"[green]Added preference: {key} = {value}[/green]")
        return

    if add_lesson:
        memory.add_lesson(add_lesson)
        console.print(f"[green]Added lesson: {add_lesson}[/green]")
        return

    if show or not any([clear, add_pref, add_lesson]):
        console.print(Panel(
            memory.get_context_summary() or "No memory yet.",
            title="Project Memory",
            border_style="blue",
        ))


@app.command()
def safety(
    file: Path = typer.Argument(..., help="File to check"),
    code: Optional[str] = typer.Option(None, "--code", "-c", help="Code string to check"),
):
    """Run safety checks on a file."""
    result = run_safety_check(str(file), code)

    if result.passed:
        console.print("[green]✓ Safety checks passed[/green]")
    else:
        console.print("[red]✗ Safety checks failed[/red]")

    if result.errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in result.errors:
            console.print(f"  - {err}")

    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warn in result.warnings:
            console.print(f"  - {warn}")

    if result.blocked_files:
        console.print("\n[bold red]Blocked Files:[/bold red]")
        for f in result.blocked_files:
            console.print(f"  - {f}")

    if not result.passed:
        raise typer.Exit(1)


@app.command()
def health():
    """Check LLM endpoint health."""
    orchestrator = create_orchestrator()
    health = asyncio.run(orchestrator.check_llm_health())

    table = Table(title="LLM Endpoint Health")
    table.add_column("Model", style="cyan")
    table.add_column("Endpoint", style="dim")
    table.add_column("Status", style="green")

    endpoints = {
        "ministral": "http://localhost:8001/v1",
        "deepseek": "http://localhost:8002/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }

    for model, status in health.items():
        table.add_row(model, endpoints.get(model, "N/A"), "✓ Healthy" if status else "✗ Unreachable")

    console.print(table)


@app.command()
def init(
    project_root: Optional[Path] = typer.Option(None, "--project", "-p", help="Project root directory"),
):
    """Initialize three-brain in a project."""
    root = project_root or Path.cwd()

    # Create .env from example
    env_example = Path(__file__).parent.parent / ".env.example"
    env_file = root / ".env"

    if env_example.exists() and not env_file.exists():
        import shutil
        shutil.copy(env_example, env_file)
        console.print(f"[green]Created .env from template at {env_file}[/green]")
        console.print("[yellow]Edit .env with your OPENROUTER_API_KEY[/yellow]")
    elif env_file.exists():
        console.print("[yellow].env already exists[/yellow]")
    else:
        console.print("[red].env.example not found[/red]")

    # Create directories
    for dir_name in ["tasks", "memory", "logs"]:
        (root / dir_name).mkdir(exist_ok=True)

    console.print(f"[green]Initialized three-brain in {root}[/green]")


# FastAPI models for serve command
class TaskRequest(BaseModel):
    task: str
    project_root: Optional[str] = None
    notes: str = ""
    auto_approve: bool = False
    no_red_team: bool = False
    files: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    final_plan: Optional[str] = None
    requires_approval: bool
    builder_output: Optional[str] = None
    analyst_output: Optional[str] = None
    strategist_output: Optional[str] = None
    red_team_output: Optional[str] = None


# FastAPI app for serve command
if FASTAPI_AVAILABLE:
    api_app = FastAPI(title="Three-Brain AI Orchestrator API")

    @api_app.post("/tasks", response_model=TaskResponse)
    async def create_task(request: TaskRequest):
        """Submit a task through the three-brain pipeline."""
        root = Path(request.project_root) if request.project_root else Path.cwd()
        
        relevant_files = []
        if request.files:
            relevant_files = [Path(f.strip()) for f in request.files.split(",")]

        orchestrator = create_orchestrator(
            project_root=root,
            enable_red_team=not request.no_red_team,
        )

        try:
            result = await orchestrator.run_task(
                task_description=request.task,
                user_notes=request.notes,
                relevant_files=relevant_files,
                auto_approve=request.auto_approve,
            )
            
            return TaskResponse(
                task_id=result.task.id,
                status=result.task.status,
                final_plan=result.final_plan,
                requires_approval=result.requires_approval,
                builder_output=result.builder_output,
                analyst_output=result.analyst_output,
                strategist_output=result.strategist_output,
                red_team_output=result.red_team_output,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    class UpdateBackendsRequest(BaseModel):
        secret: str
        ministral_url: str
        deepseek_url: str

    @api_app.post("/update-backends")
    async def update_backends(request: UpdateBackendsRequest):
        """Update Ministral/DeepSeek backend URLs from a fresh Kaggle session."""
        import os
        from dotenv import set_key
        from pathlib import Path as _Path

        expected_secret = os.getenv("UPDATE_SECRET")
        if not expected_secret or request.secret != expected_secret:
            raise HTTPException(status_code=403, detail="Invalid secret")

        env_path = _Path(__file__).parent.parent / ".env"

        os.environ["MINISTRAL_BASE_URL"] = request.ministral_url
        os.environ["DEEPSEEK_BASE_URL"] = request.deepseek_url
        set_key(str(env_path), "MINISTRAL_BASE_URL", request.ministral_url)
        set_key(str(env_path), "DEEPSEEK_BASE_URL", request.deepseek_url)

        return {"status": "updated", "ministral_url": request.ministral_url, "deepseek_url": request.deepseek_url}

    @api_app.get("/health")
    async def health_check():
        """Check LLM endpoint health."""
        orchestrator = create_orchestrator()
        health = await orchestrator.check_llm_health()
        return {"health": health}

    @api_app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get task status."""
        orchestrator = create_orchestrator()
        task = orchestrator.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task.to_dict()

    @api_app.post("/tasks/{task_id}/approve")
    async def approve_task(task_id: str, notes: str = ""):
        """Approve a task."""
        orchestrator = create_orchestrator()
        task = orchestrator.approve_task(task_id, notes)
        return {"status": "approved", "task": task.to_dict()}

    @api_app.post("/tasks/{task_id}/reject")
    async def reject_task(task_id: str, notes: str):
        """Reject a task."""
        orchestrator = create_orchestrator()
        task = orchestrator.reject_task(task_id, notes)
        return {"status": "rejected", "task": task.to_dict()}

    @api_app.post("/tasks/{task_id}/revise")
    async def revise_task(task_id: str, notes: str):
        """Request revisions to a task."""
        orchestrator = create_orchestrator()
        task = orchestrator.request_revisions(task_id, notes)
        return {"status": "revision_requested", "task": task.to_dict()}


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8005, "--port", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the Three-Brain AI Orchestrator as a REST API server."""
    if not FASTAPI_AVAILABLE:
        console.print("[red]FastAPI not installed. Run: pip install fastapi uvicorn[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold]Three-Brain Orchestrator API Server[/bold]\n"
        f"Host: {host}\n"
        f"Port: {port}",
        title="Starting Server",
        border_style="blue",
    ))

    uvicorn.run(
        "three_brain_ai.cli:api_app",
        host=host,
        port=port,
        reload=reload,
    )


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
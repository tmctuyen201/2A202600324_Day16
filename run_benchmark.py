from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/real_run",
    reflexion_attempts: int = 3,
    adaptive: bool = True,
    use_mock: bool = False
) -> None:
    """
    Run benchmark evaluation on ReAct and Reflexion agents.
    
    Args:
        dataset: Path to the dataset JSON file
        out_dir: Output directory for results
        reflexion_attempts: Maximum number of attempts for Reflexion agent
        adaptive: Use adaptive max attempts based on question difficulty
        use_mock: Use mock runtime instead of real LLM (for testing)
    """
    # Set environment variable for mock mode
    if use_mock:
        os.environ["USE_MOCK_RUNTIME"] = "true"
    else:
        os.environ["USE_MOCK_RUNTIME"] = "false"
    
    print(f"[bold blue]Loading dataset from {dataset}...[/bold blue]")
    examples = load_dataset(dataset)
    print(f"[green]Loaded {len(examples)} examples[/green]")
    
    # Initialize agents
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, adaptive_max_attempts=adaptive)
    
    # Track extensions used
    extensions = [
        "structured_evaluator",
        "reflection_memory",
        "benchmark_report_json",
        "real_llm_integration"
    ]
    
    if adaptive:
        extensions.append("adaptive_max_attempts")
    
    extensions.append("memory_compression")
    
    if use_mock:
        extensions.append("mock_mode_for_autograding")
    
    # Run ReAct agent
    print(f"\n[bold yellow]Running ReAct agent on {len(examples)} examples...[/bold yellow]")
    react_records = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]ReAct", total=len(examples))
        for i, example in enumerate(examples):
            record = react.run(example)
            react_records.append(record)
            progress.update(task, advance=1, description=f"[cyan]ReAct ({i+1}/{len(examples)})")
    
    print(f"[green]✓ ReAct completed: {sum(1 for r in react_records if r.is_correct)}/{len(react_records)} correct[/green]")
    
    # Run Reflexion agent
    print(f"\n[bold yellow]Running Reflexion agent on {len(examples)} examples...[/bold yellow]")
    reflexion_records = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[magenta]Reflexion", total=len(examples))
        for i, example in enumerate(examples):
            record = reflexion.run(example)
            reflexion_records.append(record)
            progress.update(task, advance=1, description=f"[magenta]Reflexion ({i+1}/{len(examples)})")
    
    print(f"[green]✓ Reflexion completed: {sum(1 for r in reflexion_records if r.is_correct)}/{len(reflexion_records)} correct[/green]")
    
    # Save results
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    
    # Build and save report
    mode = "mock" if use_mock else "real"
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode, extensions=extensions)
    json_path, md_path = save_report(report, out_path)
    
    print(f"\n[bold green]Results saved:[/bold green]")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    
    print(f"\n[bold blue]Summary:[/bold blue]")
    print(json.dumps(report.summary, indent=2))
    
    print(f"\n[bold cyan]Run autograde with:[/bold cyan]")
    print(f"  python autograde.py --report-path {json_path}")

if __name__ == "__main__":
    app()

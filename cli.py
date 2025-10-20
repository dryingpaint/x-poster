"""CLI for Agent Tweeter."""

import asyncio
import json
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from src.core.models import GenerateRequest, ItemKind
from src.db.operations import insert_chunks, insert_item
from src.generation.embeddings import embed_batch
from src.ingestion.chunker import create_chunks_with_overlap
from src.ingestion.pdf_processor import process_pdf_file
from src.orchestrator.pipeline import run_generation_pipeline

console = Console()


@click.group()
def main():
    """Agent Tweeter - Generate evidence-based tweets with inline citations."""
    pass


@main.command()
@click.argument("prompt")
@click.option("--max-variants", default=3, help="Maximum number of tweet variants")
@click.option("--max-thread-tweets", default=6, help="Maximum tweets in thread")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def generate(prompt: str, max_variants: int, max_thread_tweets: int, output: str | None):
    """Generate tweets from a prompt."""

    async def _generate():
        request = GenerateRequest(
            prompt=prompt,
            max_variants=max_variants,
            max_thread_tweets=max_thread_tweets,
        )

        console.print(f"\n[bold blue]🚀 Generating tweets for:[/bold blue] {prompt}\n")

        response = await run_generation_pipeline(request)

        # Display results
        if response.variants:
            console.print("\n[bold green]📝 Single Tweet Variants:[/bold green]\n")
            for i, variant in enumerate(response.variants, 1):
                console.print(Panel(variant.text, title=f"Variant {i}", border_style="green"))
                if variant.citations:
                    console.print(
                        f"   Citations: {', '.join(f'[{c.n}]' for c in variant.citations)}\n"
                    )

        if response.thread:
            console.print("\n[bold cyan]🧵 Thread:[/bold cyan]\n")
            for i, tweet in enumerate(response.thread, 1):
                console.print(Panel(tweet.text, title=f"Tweet {i}", border_style="cyan"))

        if response.sources:
            console.print(f"\n[bold yellow]📚 Sources ({len(response.sources)}):[/bold yellow]\n")
            for source in response.sources:
                url_or_uri = source.url or source.source_uri or "Unknown"
                title = source.title or "Untitled"
                console.print(f"  • {title}")
                console.print(f"    {url_or_uri}\n")

        # Save to file if requested
        if output:
            output_path = Path(output)
            with open(output_path, "w") as f:
                json.dump(response.model_dump(), f, indent=2, default=str)
            console.print(f"\n[green]✅ Saved to {output_path}[/green]")

        return response

    asyncio.run(_generate())


@main.command()
@click.option(
    "--source", "-s", required=True, type=click.Path(exists=True), help="Source file or directory"
)
@click.option(
    "--kind",
    "-k",
    type=click.Choice(["pdf", "doc", "note", "image", "other"]),
    default="pdf",
    help="Item kind",
)
@click.option("--title", "-t", help="Item title (auto-detected if not provided)")
def ingest(source: str, kind: str, title: str | None):
    """Ingest documents into the database."""

    async def _ingest():
        source_path = Path(source)

        # Handle directory
        if source_path.is_dir():
            files = list(source_path.glob("**/*.pdf"))
            console.print(f"Found {len(files)} PDF files in {source_path}")

            for file_path in files:
                await _ingest_single_file(file_path, kind, title)
        else:
            await _ingest_single_file(source_path, kind, title)

    async def _ingest_single_file(file_path: Path, kind: str, title: str | None):
        console.print(f"\n[bold]Processing:[/bold] {file_path.name}")

        try:
            # Process PDF
            if kind == "pdf":
                text, metadata = await process_pdf_file(file_path)
                final_title = title or metadata.get("title") or file_path.stem
            else:
                # For other types, just read text
                text = file_path.read_text()
                metadata = {}
                final_title = title or file_path.stem

            # Insert item
            item_id = await insert_item(
                kind=ItemKind(kind),
                title=final_title,
                source_uri=f"file://{file_path.absolute()}",
                content_text=text,
                meta=metadata,
            )

            console.print(f"   Created item: {item_id}")

            # Create chunks
            chunks_data = create_chunks_with_overlap(text, metadata)
            console.print(f"   Created {len(chunks_data)} chunks")

            # Embed chunks
            chunk_texts = [c["content"] for c in chunks_data]
            embeddings = embed_batch(chunk_texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(chunks_data, embeddings, strict=False):
                chunk["embedding"] = embedding

            # Insert chunks
            chunk_ids = await insert_chunks(item_id, chunks_data)
            console.print(f"   Inserted {len(chunk_ids)} chunks into database")

            console.print(f"[green]✅ Successfully ingested {file_path.name}[/green]")

        except Exception as e:
            console.print(f"[red]❌ Failed to ingest {file_path.name}: {e}[/red]")

    asyncio.run(_ingest())


@main.command()
def test_connection():
    """Test database and API connections."""

    async def _test():
        from src.core.config import get_config
        from src.db.client import get_db_client

        config = get_config()
        console.print("\n[bold]Testing connections...[/bold]\n")

        # Test Supabase
        try:
            db = get_db_client()
            client = db.get_client()
            result = client.table("items").select("count").execute()
            console.print("[green]✅ Supabase connection: OK[/green]")
        except Exception as e:
            console.print(f"[red]❌ Supabase connection failed: {e}[/red]")

        # Test OpenAI
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=config.openai_api_key)
            await client.models.list()
            console.print("[green]✅ OpenAI API: OK[/green]")
        except Exception as e:
            console.print(f"[red]❌ OpenAI API failed: {e}[/red]")

        # Test EXA
        if config.exa_api_key:
            try:
                from exa_py import Exa

                exa = Exa(api_key=config.exa_api_key)
                console.print("[green]✅ EXA API: OK[/green]")
            except Exception as e:
                console.print(f"[red]❌ EXA API failed: {e}[/red]")

        # Test Serper
        if config.serper_api_key:
            console.print("[green]✅ Serper API: Configured[/green]")

        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"  Embedding model: {config.openai_embedding_model} (dim={config.embedding_dim})")
        console.print(f"  Reranker provider: {config.reranker_provider}")
        console.print(f"  Primary search: {config.primary_search_provider}")

    asyncio.run(_test())


@main.group()
def db():
    """Database migration commands using Supabase CLI."""
    pass


@db.command()
def start():
    """Start local Supabase development environment."""
    console.print("\n[bold blue]🚀 Starting local Supabase...[/bold blue]\n")
    
    try:
        result = subprocess.run(
            ["npx", "supabase", "start"],
            check=True
        )
        console.print("[green]✅ Local Supabase started successfully![/green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to start local Supabase: {e}[/red]")
        raise click.ClickException("Failed to start local Supabase")


@db.command()
def stop():
    """Stop local Supabase development environment."""
    console.print("\n[bold blue]🛑 Stopping local Supabase...[/bold blue]\n")
    
    try:
        result = subprocess.run(
            ["npx", "supabase", "stop"],
            check=True
        )
        console.print("[green]✅ Local Supabase stopped successfully![/green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to stop local Supabase: {e}[/red]")
        raise click.ClickException("Failed to stop local Supabase")


@db.command()
def reset():
    """Reset local database and reapply all migrations."""
    console.print("\n[bold yellow]🔄 Resetting local database...[/bold yellow]\n")
    
    try:
        result = subprocess.run(
            ["npx", "supabase", "db", "reset"],
            check=True
        )
        console.print("[green]✅ Database reset completed successfully![/green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Database reset failed: {e}[/red]")
        raise click.ClickException("Database reset failed")


@db.command()
@click.argument("name")
def create_migration(name: str):
    """Create a new migration file."""
    console.print(f"\n[bold blue]📝 Creating migration: {name}[/bold blue]\n")
    
    try:
        result = subprocess.run(
            ["npx", "supabase", "migration", "new", name],
            check=True
        )
        console.print("[green]✅ Migration file created successfully![/green]")
        console.print("[yellow]💡 Edit the migration file in supabase/migrations/ before applying![/yellow]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to create migration: {e}[/red]")


@db.command()
@click.argument("name") 
def diff(name: str):
    """Generate migration from local database changes."""
    console.print(f"\n[bold blue]📊 Generating diff migration: {name}[/bold blue]\n")
    
    try:
        result = subprocess.run(
            ["npx", "supabase", "db", "diff", "-f", name],
            check=True
        )
        console.print("[green]✅ Diff migration generated successfully![/green]")
        console.print("[yellow]💡 Review the generated SQL before deploying![/yellow]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to generate diff: {e}[/red]")


@db.command()
@click.option("--include-seed", is_flag=True, help="Include seed data")
def push(include_seed: bool):
    """Push migrations to remote Supabase project."""
    console.print("\n[bold blue]🚀 Pushing migrations to remote...[/bold blue]\n")
    
    cmd = ["npx", "supabase", "db", "push"]
    if include_seed:
        cmd.append("--include-seed")
    
    try:
        result = subprocess.run(cmd, check=True)
        console.print("[green]✅ Migrations pushed successfully![/green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to push migrations: {e}[/red]")
        raise click.ClickException("Migration push failed")


@db.command()
@click.argument("project_ref")
def link(project_ref: str):
    """Link to a Supabase project."""
    console.print(f"\n[bold blue]🔗 Linking to project: {project_ref}[/bold blue]\n")
    
    try:
        result = subprocess.run(
            ["npx", "supabase", "link", "--project-ref", project_ref],
            check=True
        )
        console.print("[green]✅ Project linked successfully![/green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to link project: {e}[/red]")


if __name__ == "__main__":
    main()

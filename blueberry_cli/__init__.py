"""Typer CLI utilities for Blueberry LLM experimentation."""

from .cli import app


def main() -> None:
    app()


__all__ = ["app", "main"]

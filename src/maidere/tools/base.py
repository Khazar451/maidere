"""Base tool interface for Maidere tools."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseTool(ABC, BaseModel):
    """Abstract base class for Maidere agent tools."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for tool parameters
    is_concurrent_safe: bool = False  # Fail-closed default: tools are serial unless opted in

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool action asynchronously and return string output.

        Each tool declares its own named parameters (matching its JSON schema), so
        the base signature is deliberately open: ``(*args, **kwargs)`` lets mypy
        accept those narrower overrides.
        """

    def to_ollama_schema(self) -> dict[str, Any]:
        """Convert tool definition to Ollama API tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

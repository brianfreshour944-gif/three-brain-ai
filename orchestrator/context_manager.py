"""Context manager for building agent context."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.memory import get_memory_manager


@dataclass
class TaskContext:
    """Context for a single task."""
    task_id: str
    task_description: str
    project_root: Path
    relevant_files: List[Path] = field(default_factory=list)
    file_contents: Dict[str, str] = field(default_factory=dict)
    user_notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Builds and manages context for agents."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.memory = get_memory_manager()
        self._file_cache: Dict[str, str] = {}

    def build_context(self, task: TaskContext) -> str:
        """Build full context string for agents."""
        parts = []

        # Project memory
        memory_summary = self.memory.get_context_summary()
        if memory_summary != "No project memory yet.":
            parts.append("=== PROJECT MEMORY ===")
            parts.append(memory_summary)

        # Task description
        parts.append("\n=== TASK ===")
        parts.append(task.task_description)

        # User notes
        if task.user_notes:
            parts.append("\n=== USER NOTES ===")
            parts.append(task.user_notes)

        # Relevant files
        if task.relevant_files:
            parts.append("\n=== RELEVANT FILES ===")
            for file_path in task.relevant_files:
                content = self._get_file_content(file_path)
                if content:
                    rel_path = file_path.relative_to(self.project_root) if file_path.is_absolute() else file_path
                    parts.append(f"\n--- {rel_path} ---")
                    parts.append(content)

        # Previously provided file contents
        if task.file_contents:
            parts.append("\n=== PROVIDED FILE CONTENTS ===")
            for path, content in task.file_contents.items():
                parts.append(f"\n--- {path} ---")
                parts.append(content)

        return "\n".join(parts)

    def _get_file_content(self, file_path: Path) -> Optional[str]:
        """Get file content with caching."""
        cache_key = str(file_path)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        try:
            full_path = file_path if file_path.is_absolute() else self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                # Skip large files
                if full_path.stat().st_size > 100_000:
                    return f"[File too large to display: {full_path.stat().st_size} bytes]"

                content = full_path.read_text(encoding="utf-8", errors="replace")
                self._file_cache[cache_key] = content
                return content
        except Exception:
            pass
        return None

    def find_relevant_files(self, task_description: str, max_files: int = 10) -> List[Path]:
        """Find files relevant to the task (simple keyword matching)."""
        keywords = self._extract_keywords(task_description)
        relevant = []

        for ext in [".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md", ".txt"]:
            for file_path in self.project_root.rglob(f"*{ext}"):
                if self._should_skip(file_path):
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace").lower()
                    score = sum(1 for kw in keywords if kw in content)
                    if score > 0:
                        relevant.append((score, file_path))
                except Exception:
                    continue

        relevant.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in relevant[:max_files]]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from task description."""
        # Simple extraction - could be enhanced with NLP
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                     "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
                     "been", "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "can", "this",
                     "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"}

        words = text.lower().replace(".", " ").replace(",", " ").replace("(", " ").replace(")", " ").split()
        return [w for w in words if len(w) > 3 and w not in stopwords][:20]

    def _should_skip(self, path: Path) -> bool:
        """Check if file should be skipped."""
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
                     "dist", "build", ".pytest_cache", ".mypy_cache", ".coverage"}
        return any(part in skip_dirs for part in path.parts)

    def clear_cache(self) -> None:
        """Clear file cache."""
        self._file_cache.clear()
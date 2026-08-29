"""Memory layer for project context and preferences using Markdown files."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.config import MEMORY_DIR


@dataclass
class ProjectMemory:
    """Persistent project memory."""
    preferences: Dict[str, str] = field(default_factory=dict)
    architecture_notes: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    approved_solutions: List[Dict[str, Any]] = field(default_factory=list)
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryManager:
    """Manages project memory persistence using Markdown files."""

    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = memory_dir or Path(MEMORY_DIR)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory: Optional["ProjectMemory"] = None
        self._load()

    def _load(self) -> None:
        """Load memory from Markdown files."""
        self._memory = ProjectMemory()

        # Load preferences
        prefs_file = self.memory_dir / "user_preferences.md"
        if prefs_file.exists():
            self._parse_preferences(prefs_file.read_text())

        # Load architecture decisions
        arch_file = self.memory_dir / "architecture_decisions.md"
        if arch_file.exists():
            self._parse_architecture_notes(arch_file.read_text())

        # Load lessons learned
        lessons_file = self.memory_dir / "lessons.md"
        if lessons_file.exists():
            self._parse_lessons(lessons_file.read_text())

        # Load approved solutions
        approved_file = self.memory_dir / "approved_solutions.md"
        if approved_file.exists():
            self._parse_approved_solutions(approved_file.read_text())

        # Load failed attempts
        failed_file = self.memory_dir / "failed_attempts.md"
        if failed_file.exists():
            self._parse_failed_attempts(failed_file.read_text())

        # Also load legacy JSON if exists (for backward compatibility)
        legacy_file = Path(MEMORY_DIR) / "project_memory.json"
        if legacy_file.exists() and not any([
            (self.memory_dir / "user_preferences.md").exists(),
            (self.memory_dir / "architecture_decisions.md").exists(),
            (self.memory_dir / "lessons.md").exists(),
        ]):
            self._load_legacy_json(legacy_file)

    def _parse_preferences(self, content: str) -> None:
        """Parse user preferences from Markdown."""
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                key, value = line[2:].split(":", 1)
                self._memory.preferences[key.strip()] = value.strip()

    def _parse_architecture_notes(self, content: str) -> None:
        """Parse architecture notes from Markdown."""
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                self._memory.architecture_notes.append(line[2:])

    def _parse_lessons(self, content: str) -> None:
        """Parse lessons learned from Markdown."""
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                self._memory.lessons_learned.append(line[2:])

    def _parse_approved_solutions(self, content: str) -> None:
        """Parse approved solutions from Markdown."""
        # Simple parsing - each solution starts with ## 
        sections = content.split("## ")
        for section in sections[1:]:
            lines = section.strip().split("\n")
            if lines:
                title = lines[0].strip()
                content = "\n".join(lines[1:]).strip()
                self._memory.approved_solutions.append({
                    "title": title,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                })

    def _parse_failed_attempts(self, content: str) -> None:
        """Parse failed attempts from Markdown."""
        sections = content.split("## ")
        for section in sections[1:]:
            lines = section.strip().split("\n")
            if lines:
                title = lines[0].strip()
                content_text = "\n".join(lines[1:]).strip()
                self._memory.failed_attempts.append({
                    "title": title,
                    "content": content_text,
                    "timestamp": datetime.now().isoformat(),
                })

    def _load_legacy_json(self, file_path: Path) -> None:
        """Load from legacy JSON format."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self._memory = ProjectMemory(**data)
        except Exception:
            pass

    def _save_all(self) -> None:
        """Save all memory to Markdown files."""
        self._save_preferences()
        self._save_architecture_notes()
        self._save_lessons()
        self._save_approved_solutions()
        self._save_failed_attempts()

    def _save_preferences(self) -> None:
        """Save preferences to Markdown."""
        lines = ["# User Preferences\n"]
        for key, value in sorted(self._memory.preferences.items()):
            lines.append(f"- {key}: {value}")
        (self.memory_dir / "user_preferences.md").write_text("\n".join(lines))

    def _save_architecture_notes(self) -> None:
        """Save architecture notes to Markdown."""
        lines = ["# Architecture Decisions\n"]
        for note in self._memory.architecture_notes:
            lines.append(f"- {note}")
        (self.memory_dir / "architecture_decisions.md").write_text("\n".join(lines))

    def _save_lessons(self) -> None:
        """Save lessons learned to Markdown."""
        lines = ["# Lessons Learned\n"]
        for lesson in self._memory.lessons_learned:
            lines.append(f"- {lesson}")
        (self.memory_dir / "lessons.md").write_text("\n".join(lines))

    def _save_approved_solutions(self) -> None:
        """Save approved solutions to Markdown."""
        lines = ["# Approved Solutions\n"]
        for sol in self._memory.approved_solutions:
            lines.append(f"## {sol.get('title', 'Untitled')}")
            lines.append(sol.get("content", ""))
            lines.append("")
        (self.memory_dir / "approved_solutions.md").write_text("\n".join(lines))

    def _save_failed_attempts(self) -> None:
        """Save failed attempts to Markdown."""
        lines = ["# Failed Attempts\n"]
        for fail in self._memory.failed_attempts:
            lines.append(f"## {fail.get('title', 'Untitled')}")
            lines.append(fail.get("content", ""))
            lines.append("")
        (self.memory_dir / "failed_attempts.md").write_text("\n".join(lines))

    @property
    def memory(self) -> "ProjectMemory":
        return self._memory or ProjectMemory()

    def add_preference(self, key: str, value: str) -> None:
        """Add a user preference."""
        self._memory.preferences[key] = value
        self._save_preferences()

    def add_architecture_note(self, note: str) -> None:
        """Add an architecture decision note."""
        self._memory.architecture_notes.append(f"[{datetime.now().isoformat()}] {note}")
        self._save_architecture_notes()

    def add_lesson(self, lesson: str) -> None:
        """Add a lesson learned."""
        self._memory.lessons_learned.append(f"[{datetime.now().isoformat()}] {lesson}")
        self._save_lessons()

    def add_decision(self, decision: str, rationale: str, context: str = "") -> None:
        """Record a decision."""
        self._memory.decisions.append({
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "rationale": rationale,
            "context": context,
        })
        # Also save as architecture note
        self.add_architecture_note(f"Decision: {decision} - {rationale}")

    def add_approved_solution(self, title: str, content: str) -> None:
        """Record an approved solution."""
        self._memory.approved_solutions.append({
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_approved_solutions()

    def add_failed_attempt(self, title: str, content: str) -> None:
        """Record a failed attempt."""
        self._memory.failed_attempts.append({
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_failed_attempts()

    def get_context_summary(self) -> str:
        """Get a summary of memory for context injection."""
        parts = []

        if self._memory.preferences:
            parts.append("## User Preferences\n" + "\n".join(
                f"- {k}: {v}" for k, v in self._memory.preferences.items()
            ))

        if self._memory.architecture_notes:
            parts.append("## Architecture Notes\n" + "\n".join(
                f"- {note}" for note in self._memory.architecture_notes[-10:]
            ))

        if self._memory.lessons_learned:
            parts.append("## Lessons Learned\n" + "\n".join(
                f"- {lesson}" for lesson in self._memory.lessons_learned[-10:]
            ))

        if self._memory.approved_solutions:
            parts.append("## Approved Solutions\n" + "\n".join(
                f"- {s.get('title', 'Untitled')}" for s in self._memory.approved_solutions[-5:]
            ))

        if self._memory.failed_attempts:
            parts.append("## Failed Attempts\n" + "\n".join(
                f"- {f.get('title', 'Untitled')}" for f in self._memory.failed_attempts[-5:]
            ))

        if self._memory.decisions:
            parts.append("## Recent Decisions\n" + "\n".join(
                f"- {d['timestamp']}: {d['decision']} ({d['rationale']})"
                for d in self._memory.decisions[-5:]
            ))

        return "\n\n".join(parts) if parts else "No project memory yet."

    def clear(self) -> None:
        """Clear all memory."""
        self._memory = ProjectMemory()
        for fname in ["user_preferences.md", "architecture_decisions.md", "lessons.md",
                      "approved_solutions.md", "failed_attempts.md"]:
            fpath = self.memory_dir / fname
            if fpath.exists():
                fpath.unlink()


def get_memory_manager() -> "MemoryManager":
    """Get singleton memory manager."""
    return MemoryManager()
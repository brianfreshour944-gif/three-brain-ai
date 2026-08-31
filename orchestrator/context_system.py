"""Context system with Repomix integration and token counting."""

from __future__ import annotations

import os
import subprocess
import tiktoken
from pathlib import Path
from typing import Dict, List, Optional, Set

from orchestrator.config import MINISTRAL_CTX, DEEPSEEK_CTX, STRATEGIST_CTX


# Model context windows (in tokens)
MODEL_CONTEXT_WINDOWS = {
    "ministral": MINISTRAL_CTX,
    "deepseek": DEEPSEEK_CTX,
    "strategist": STRATEGIST_CTX,
}

# Default encoding for token counting
DEFAULT_ENCODING = "cl100k_base"  # GPT-4 / Claude encoding


class TokenCounter:
    """Token counting utilities for context window management."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of chat messages."""
        # Rough approximation: 4 tokens per message overhead + content tokens
        total = 0
        for msg in messages:
            total += 4  # message overhead
            total += self.count_tokens(msg.get("content", ""))
        return total

    def get_model_limit(self, model: str) -> int:
        """Get context window limit for a model."""
        return MODEL_CONTEXT_WINDOWS.get(model, 8192)

    def check_fit(
        self,
        context: str,
        model: str,
        reserve_tokens: int = 1000,
    ) -> tuple[bool, int, int]:
        """Check if context fits in model's context window."""
        tokens = self.count_tokens(context)
        limit = self.get_model_limit(model)
        available = limit - reserve_tokens
        return tokens <= available, tokens, available


class RepomixManager:
    """Manages Repomix for intelligent code context packing."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.repomix_path = self._find_repomix()

    def _find_repomix(self) -> Optional[str]:
        """Find repomix executable."""
        # Check if installed via npx
        try:
            result = subprocess.run(
                ["npx", "repomix", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return "npx"
        except Exception:
            pass
        return None

    def is_available(self) -> bool:
        """Check if Repomix is available."""
        return self.repomix_path is not None

    def pack_repository(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        output_file: Optional[Path] = None,
    ) -> str:
        """Pack repository context using Repomix."""
        if not self.is_available():
            raise RuntimeError("Repomix not available. Install with: npm install -g repomix")

        cmd = ["npx", "repomix", "--style", "markdown", "--compress"]

        if include_patterns:
            for pattern in include_patterns:
                cmd.extend(["--include", pattern])

        if exclude_patterns:
            for pattern in exclude_patterns:
                cmd.extend(["--ignore", pattern])

        if output_file:
            cmd.extend(["--output", str(output_file)])
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and output_file.exists():
                return output_file.read_text()
            raise RuntimeError(f"Repomix failed: {result.stderr}")
        else:
            cmd.extend(["--stdout"])
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout
            raise RuntimeError(f"Repomix failed: {result.stderr}")


class ContextBuilder:
    """Builds optimized context for agents with token management."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.token_counter = TokenCounter()
        self.repomix = RepomixManager(project_root)

    def build_context(
        self,
        task_description: str,
        user_notes: str = "",
        relevant_files: Optional[List[Path]] = None,
        file_contents: Optional[Dict[str, str]] = None,
        target_model: str = "strategist",
        max_tokens: Optional[int] = None,
        use_repomix: bool = True,
        reserve_tokens: int = 2000,
    ) -> str:
        """Build optimized context for a target model."""

        # Determine token budget
        model_limit = self.token_counter.get_model_limit(target_model)
        token_budget = max_tokens or (model_limit - reserve_tokens)  # Reserve tokens for response

        # Build base context
        parts = []

        # Task description (always included)
        parts.append(f"# TASK\n{task_description}")

        # User notes
        if user_notes:
            parts.append(f"\n# USER NOTES\n{user_notes}")

        # Project memory
        from orchestrator.memory import get_memory_manager
        memory = get_memory_manager()
        memory_summary = memory.get_context_summary()
        if memory_summary != "No project memory yet.":
            parts.append(f"\n# PROJECT MEMORY\n{memory_summary}")

        # Trading performance stats (real, pre-computed — never ask the LLM
        # to calculate Sharpe/drawdown/etc. itself, they are unreliable at
        # multi-step arithmetic; always feed the already-computed numbers in).
        try:
            import sys
            sys.path.insert(0, "/home/ubuntu/Apex_oracle_bot/src")
            from db import get_latest_experiment_stats
            stats = get_latest_experiment_stats()
            if stats:
                stats_text = (
                    f"Experiment: {stats['experiment_id']}\n"
                    f"Sharpe ratio: {stats['sharpe']:.2f}\n"
                    f"Max drawdown: {stats['max_drawdown_pct']:.1f}%\n"
                    f"Total return: {stats['total_return_pct']:.1f}%\n"
                    f"Profit factor: {stats['profit_factor']:.2f}\n"
                    f"Status: {stats['status']}\n"
                    f"Computed at: {stats['computed_at']}"
                )
                parts.append(f"\n# LATEST TRADING PERFORMANCE STATS (pre-computed, treat as fact)\n{stats_text}")
        except Exception:
            pass  # non-fatal: proceed without stats if unavailable

        # Relevant files (explicit)
        if relevant_files:
            parts.append("\n# RELEVANT FILES")
            for file_path in relevant_files:
                content = self._get_file_content(file_path)
                if content:
                    rel_path = file_path.relative_to(self.project_root) if file_path.is_absolute() else file_path
                    parts.append(f"\n## {rel_path}\n```\n{content}\n```")

        # Provided file contents
        if file_contents:
            parts.append("\n# PROVIDED FILE CONTENTS")
            for path, content in file_contents.items():
                parts.append(f"\n## {path}\n```\n{content}\n```")

        # Repomix for additional context (if available and budget allows)
        if use_repomix and self.repomix.is_available():
            repomix_context = self._get_repomix_context(
                token_budget - self._estimate_tokens("\n".join(parts)),
                relevant_files,
            )
            if repomix_context:
                parts.append(f"\n# REPOSITORY CONTEXT (Repomix)\n{repomix_context}")

        full_context = "\n".join(parts)

        # Trim if over budget
        if self.token_counter.count_tokens(full_context) > token_budget:
            full_context = self._trim_context(full_context, token_budget)

        return full_context

    def _estimate_tokens(self, text: str) -> int:
        return self.token_counter.count_tokens(text)

    def _get_file_content(self, file_path: Path) -> Optional[str]:
        """Get file content with size limit."""
        try:
            full_path = file_path if file_path.is_absolute() else self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                size = full_path.stat().st_size
                if size > 50000:  # 50KB limit per file
                    return f"[File too large: {size} bytes]"
                return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        return None

    def _get_repomix_context(self, available_tokens: int, relevant_files: Optional[List[Path]] = None) -> Optional[str]:
        """Get Repomix context within token budget."""
        if available_tokens < 1000:
            return None

        try:
            exclude = [
                "*.pyc", "__pycache__", ".git", "node_modules",
                "venv", ".venv", "dist", "build", "*.log",
                "*.lock", "*.md", "*.txt", "*.json", "*.yaml", "*.yml",
            ]
            # Add relevant files to include
            include = []
            if relevant_files:
                for f in relevant_files:
                    include.append(str(f.relative_to(self.project_root) if f.is_absolute() else f))

            return self.repomix.pack_repository(
                include_patterns=include if include else None,
                exclude_patterns=exclude,
                max_tokens=available_tokens,
            )
        except Exception:
            return None

    def _trim_context(self, context: str, max_tokens: int) -> str:
        """Trim context to fit token budget by removing less important sections."""
        tokens = self.token_counter.count_tokens(context)
        if tokens <= max_tokens:
            return context

        # Simple trim: keep first and last portions
        lines = context.split("\n")
        # Keep task, user notes, memory (first ~30%), and last portions
        keep_start = int(len(lines) * 0.3)
        keep_end = int(len(lines) * 0.1)

        trimmed = "\n".join(lines[:keep_start] + ["\n...[trimmed]...\n"] + lines[-keep_end:])

        if self.token_counter.count_tokens(trimmed) > max_tokens:
            # Aggressive trim
            trimmed = "\n".join(lines[:100] + ["\n...[heavily trimmed]...\n"] + lines[-50:])

        return trimmed


def create_context_builder(project_root: Path) -> ContextBuilder:
    """Factory to create context builder."""
    return ContextBuilder(project_root)
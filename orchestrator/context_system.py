"""Context system with Repomix integration and token counting."""

from __future__ import annotations

import asyncio
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
    
    # Shared encoding instance to avoid recreating
    _encoding_cache: Dict[str, tiktoken.Encoding] = {}
    
    def __init__(self, encoding_name: str = DEFAULT_ENCODING):
        if encoding_name not in TokenCounter._encoding_cache:
            TokenCounter._encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        self.encoding = TokenCounter._encoding_cache[encoding_name]
    
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


class CachedRepomixManager:
    """Manages Repomix with intelligent caching based on file mtimes."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.repomix_path = self._find_repomix()
        self._cache: Optional[str] = None
        self._cache_mtime: float = 0
        self._cache_include_patterns: Optional[List[str]] = None
        self._cache_exclude_patterns: Optional[List[str]] = None
        self._lock = asyncio.Lock()
    
    def _find_repomix(self) -> Optional[str]:
        """Find repomix executable."""
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
    
    def _get_latest_mtime(self, include_patterns: Optional[List[str]] = None) -> float:
        """Get the latest modification time of tracked files."""
        max_mtime = 0
        extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", 
                     ".toml", ".ini", ".cfg", ".conf", ".md", ".txt", ".html", ".css",
                     ".sh", ".bash", ".zsh", ".dockerfile", ".dockerignore"]
        
        for ext in extensions:
            try:
                for f in self.project_root.rglob(f"*{ext}"):
                    if self._should_skip_file(f, include_patterns):
                        continue
                    try:
                        mtime = f.stat().st_mtime
                        if mtime > max_mtime:
                            max_mtime = mtime
                    except OSError:
                        pass
            except Exception:
                pass
        return max_mtime
    
    def _should_skip_file(self, file_path: Path, include_patterns: Optional[List[str]]) -> bool:
        """Check if file should be skipped based on common ignore patterns."""
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
                     "dist", "build", ".pytest_cache", ".mypy_cache", ".coverage",
                     ".idea", ".vscode", "*.egg-info", "htmlcov", ".tox"}
        
        # Skip if in ignored directory
        for part in file_path.parts:
            if part in skip_dirs:
                return True
        
        # If include_patterns specified, only include matching files
        if include_patterns:
            import fnmatch
            rel_path = str(file_path.relative_to(self.project_root))
            matched = any(fnmatch.fnmatch(rel_path, pat) for pat in include_patterns)
            if not matched:
                return True
        
        return False
    
    def _cache_key(self, include_patterns: Optional[List[str]], 
                   exclude_patterns: Optional[List[str]]) -> str:
        """Generate cache key from patterns."""
        return f"{tuple(include_patterns or [])}:{tuple(exclude_patterns or [])}"
    
    async def get_context(
        self,
        available_tokens: int,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Get cached Repomix context or generate new one if files changed."""
        if available_tokens < 1000:
            return None
        
        if not self.is_available():
            return None
        
        cache_key = self._cache_key(include_patterns, exclude_patterns)
        
        # Quick check: if cache exists and patterns match, verify mtime
        if (self._cache is not None and 
            self._cache_include_patterns == include_patterns and
            self._cache_exclude_patterns == exclude_patterns):
            
            current_mtime = self._get_latest_mtime(include_patterns)
            if current_mtime <= self._cache_mtime:
                # Return trimmed cached version
                return self._trim_to_tokens(self._cache, available_tokens)
        
        # Need to regenerate - acquire lock
        async with self._lock:
            # Double-check after acquiring lock
            if (self._cache is not None and 
                self._cache_include_patterns == include_patterns and
                self._cache_exclude_patterns == exclude_patterns):
                
                current_mtime = self._get_latest_mtime(include_patterns)
                if current_mtime <= self._cache_mtime:
                    return self._trim_to_tokens(self._cache, available_tokens)
            
            # Generate new context
            try:
                exclude = [
                    "*.pyc", "__pycache__", ".git", "node_modules",
                    "venv", ".venv", "dist", "build", "*.log",
                    "*.lock", "*.md", "*.txt", "*.json", "*.yaml", "*.yml",
                ]
                if exclude_patterns:
                    exclude.extend(exclude_patterns)
                
                cmd = ["npx", "repomix", "--style", "markdown", "--compress", "--stdout"]
                
                if include_patterns:
                    for pattern in include_patterns:
                        cmd.extend(["--include", pattern])
                
                for pattern in exclude:
                    cmd.extend(["--ignore", pattern])
                
                # Run in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                )
                
                if result.returncode == 0:
                    self._cache = result.stdout
                    self._cache_mtime = self._get_latest_mtime(include_patterns)
                    self._cache_include_patterns = include_patterns
                    self._cache_exclude_patterns = exclude_patterns
                    return self._trim_to_tokens(self._cache, available_tokens)
                else:
                    # Log error but don't fail - return None to continue without repomix
                    return None
                    
            except Exception:
                return None
    
    def _trim_to_tokens(self, text: str, max_tokens: int) -> str:
        """Trim text to fit within token budget."""
        counter = TokenCounter()
        tokens = counter.count_tokens(text)
        if tokens <= max_tokens:
            return text
        
        # Trim proportionally
        ratio = max_tokens / tokens
        char_limit = int(len(text) * ratio * 0.95)  # Conservative estimate
        return text[:char_limit] + "\n...[trimmed to fit token budget]..."
    
    def invalidate_cache(self):
        """Force cache invalidation."""
        self._cache = None
        self._cache_mtime = 0


class ContextBuilder:
    """Builds optimized context for agents with token management."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.token_counter = TokenCounter()
        self.repomix = CachedRepomixManager(project_root)
    
    async def build_context(
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
        """Build optimized context for a target model (async with caching)."""
        
        # Determine token budget
        model_limit = self.token_counter.get_model_limit(target_model)
        token_budget = max_tokens or (model_limit - reserve_tokens)
        
        # Build base context (without repomix)
        base_parts = []
        base_parts.append(f"# TASK\n{task_description}")
        
        if user_notes:
            base_parts.append(f"\n# USER NOTES\n{user_notes}")
        
        # Project memory
        from orchestrator.memory import get_memory_manager
        memory = get_memory_manager()
        memory_summary = memory.get_context_summary()
        if memory_summary != "No project memory yet.":
            base_parts.append(f"\n# PROJECT MEMORY\n{memory_summary}")
        
        if relevant_files:
            base_parts.append("\n# RELEVANT FILES")
            for file_path in relevant_files:
                content = self._get_file_content(file_path)
                if content:
                    rel_path = (file_path.relative_to(self.project_root) 
                               if file_path.is_absolute() else file_path)
                    base_parts.append(f"\n## {rel_path}\n```\n{content}\n```")
        
        if file_contents:
            base_parts.append("\n# PROVIDED FILE CONTENTS")
            for path, content in file_contents.items():
                base_parts.append(f"\n## {path}\n```\n{content}\n```")
        
        base_context = "\n".join(base_parts)
        base_tokens = self.token_counter.count_tokens(base_context)
        
        # Get Repomix context asynchronously (with caching)
        repomix_context = ""
        if use_repomix and self.repomix.is_available():
            available_for_repomix = token_budget - base_tokens - 500  # Reserve for overhead
            if available_for_repomix > 1000:
                include_patterns = None
                if relevant_files:
                    include_patterns = [
                        str(f.relative_to(self.project_root) if f.is_absolute() else f)
                        for f in relevant_files
                    ]
                
                repomix_context = await self.repomix.get_context(
                    available_tokens=available_for_repomix,
                    include_patterns=include_patterns,
                ) or ""
        
        full_context = base_context + repomix_context
        
        # Final trim if over budget
        if self.token_counter.count_tokens(full_context) > token_budget:
            full_context = self._trim_context(full_context, token_budget)
        
        return full_context
    
    def build_context_sync(
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
        """Synchronous version of build_context for backward compatibility."""
        
        # Determine token budget
        model_limit = self.token_counter.get_model_limit(target_model)
        token_budget = max_tokens or (model_limit - reserve_tokens)
        
        # Build base context
        parts = []
        parts.append(f"# TASK\n{task_description}")
        
        if user_notes:
            parts.append(f"\n# USER NOTES\n{user_notes}")
        
        from orchestrator.memory import get_memory_manager
        memory = get_memory_manager()
        memory_summary = memory.get_context_summary()
        if memory_summary != "No project memory yet.":
            parts.append(f"\n# PROJECT MEMORY\n{memory_summary}")
        
        if relevant_files:
            parts.append("\n# RELEVANT FILES")
            for file_path in relevant_files:
                content = self._get_file_content(file_path)
                if content:
                    rel_path = (file_path.relative_to(self.project_root) 
                               if file_path.is_absolute() else file_path)
                    parts.append(f"\n## {rel_path}\n```\n{content}\n```")
        
        if file_contents:
            parts.append("\n# PROVIDED FILE CONTENTS")
            for path, content in file_contents.items():
                parts.append(f"\n## {path}\n```\n{content}\n```")
        
        # Repomix (synchronous fallback - no cache)
        if use_repomix and self.repomix.is_available():
            available_for_repomix = token_budget - self._estimate_tokens("\n".join(parts))
            if available_for_repomix > 1000:
                include = []
                if relevant_files:
                    for f in relevant_files:
                        rel = f.relative_to(self.project_root) if f.is_absolute() else f
                        include.append(str(rel))
                
                try:
                    repomix_ctx = self.repomix.pack_repository(
                        include_patterns=include if include else None,
                        exclude_patterns=[
                            "*.pyc", "__pycache__", ".git", "node_modules",
                            "venv", ".venv", "dist", "build", "*.log",
                            "*.lock", "*.md", "*.txt", "*.json", "*.yaml", "*.yml",
                        ],
                        max_tokens=available_for_repomix,
                    )
                    parts.append(f"\n# REPOSITORY CONTEXT (Repomix)\n{repomix_ctx}")
                except Exception:
                    pass
        
        full_context = "\n".join(parts)
        
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
                if size > 50000:
                    return f"[File too large: {size} bytes]"
                return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        return None
    
    def _trim_context(self, context: str, max_tokens: int) -> str:
        """Trim context to fit token budget by removing less important sections."""
        tokens = self.token_counter.count_tokens(context)
        if tokens <= max_tokens:
            return context
        
        lines = context.split("\n")
        keep_start = int(len(lines) * 0.3)
        keep_end = int(len(lines) * 0.1)
        
        trimmed = "\n".join(lines[:keep_start] + ["\n...[trimmed]...\n"] + lines[-keep_end:])
        
        if self.token_counter.count_tokens(trimmed) > max_tokens:
            trimmed = "\n".join(lines[:100] + ["\n...[heavily trimmed]...\n"] + lines[-50:])
        
        return trimmed


def create_context_builder(project_root: Path) -> ContextBuilder:
    """Factory to create context builder."""
    return ContextBuilder(project_root)
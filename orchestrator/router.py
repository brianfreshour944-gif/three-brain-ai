"""Task routing and complexity classification for the three-brain orchestrator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from orchestrator.config import MINISTRAL_MAX_TOKENS, DEEPSEEK_MAX_TOKENS


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"      # Fast path: single LLM call, no context building
    SIMPLE = "simple"        # Reduced pipeline: single round, limited context
    STANDARD = "standard"    # Full two-round pipeline with Repomix
    COMPLEX = "complex"      # Full pipeline + Red Team + extended context


@dataclass
class TaskClassification:
    """Result of task classification."""
    complexity: TaskComplexity
    confidence: float
    reasoning: str
    recommended_max_tokens: int
    use_repomix: bool
    skip_round2: bool
    skip_red_team: bool


class TaskClassifier:
    """Classifies task complexity to route to appropriate pipeline."""
    
    # Keywords that indicate trivial tasks
    TRIVIAL_KEYWORDS = {
        "say", "hello", "hi", "test", "ping", "echo", "print", "return",
        "add", "sum", "plus", "minus", "subtract", "multiply", "divide",
        "simple", "basic", "quick", "small", "tiny", "minimal"
    }
    
    # Keywords that indicate simple tasks
    SIMPLE_KEYWORDS = {
        "function", "method", "class", "decorator", "helper", "utility",
        "validate", "format", "parse", "convert", "transform", "filter",
        "sort", "map", "reduce", "find", "search", "get", "set", "check",
        "is_", "has_", "can_", "should_", "will_", "create", "make", "build"
    }
    
    # Keywords that indicate complex tasks
    COMPLEX_KEYWORDS = {
        "architecture", "system", "design", "refactor", "migrate", "integrate",
        "optimize", "scale", "distributed", "microservice", "database",
        "authentication", "authorization", "security", "encryption",
        "real-time", "streaming", "async", "concurrent", "parallel",
        "machine learning", "ml", "ai", "model", "training", "inference",
        "pipeline", "workflow", "orchestration", "deployment", "ci/cd",
        "monitoring", "logging", "observability", "tracing"
    }
    
    # File extensions that indicate code tasks
    CODE_EXTENSIONS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h", ".hpp"}
    
    def __init__(self):
        self.trivial_pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in self.TRIVIAL_KEYWORDS) + r")\b",
            re.IGNORECASE
        )
        self.simple_pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in self.SIMPLE_KEYWORDS) + r")\b",
            re.IGNORECASE
        )
        self.complex_pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in self.COMPLEX_KEYWORDS) + r")\b",
            re.IGNORECASE
        )
    
    def classify(self, task_description: str, relevant_files: Optional[list] = None) -> TaskClassification:
        """
        Classify task complexity.
        
        Conservative approach: default to STANDARD unless clearly TRIVIAL or COMPLEX.
        False negatives (running full pipeline on trivial task) are preferred
        over false positives (skipping analysis on complex task).
        """
        desc_lower = task_description.lower()
        word_count = len(task_description.split())
        has_files = bool(relevant_files)
        has_code_files = any(
            any(f.endswith(ext) for ext in self.CODE_EXTENSIONS) 
            for f in (relevant_files or [])
        )
        
        # Check for trivial patterns
        trivial_matches = len(self.trivial_pattern.findall(task_description))
        simple_matches = len(self.simple_pattern.findall(task_description))
        complex_matches = len(self.complex_pattern.findall(task_description))
        
        # TRIVIAL: Very short, trivial keywords, no files, simple operations
        if (word_count <= 10 and trivial_matches >= 1 and 
            not has_files and simple_matches == 0 and complex_matches == 0):
            return TaskClassification(
                complexity=TaskComplexity.TRIVIAL,
                confidence=0.9,
                reasoning=f"Trivial task detected: {word_count} words, trivial keywords={trivial_matches}, no files",
                recommended_max_tokens=512,
                use_repomix=False,
                skip_round2=True,
                skip_red_team=True
            )
        
        # COMPLEX: Complex keywords, many files, or long description
        if (complex_matches >= 2 or 
            word_count > 100 or 
            (has_files and has_code_files and word_count > 50)):
            return TaskClassification(
                complexity=TaskComplexity.COMPLEX,
                confidence=0.85,
                reasoning=f"Complex task: {complex_matches} complex keywords, {word_count} words, files={has_files}",
                recommended_max_tokens=8192,
                use_repomix=True,
                skip_round2=False,
                skip_red_team=False
            )
        
        # SIMPLE: Some simple keywords, short-medium description, maybe files
        if (simple_matches >= 1 and word_count <= 50 and 
            complex_matches == 0 and not has_code_files):
            return TaskClassification(
                complexity=TaskComplexity.SIMPLE,
                confidence=0.7,
                reasoning=f"Simple task: {simple_matches} simple keywords, {word_count} words",
                recommended_max_tokens=2048,
                use_repomix=False,
                skip_round2=True,
                skip_red_team=True
            )
        
        # DEFAULT: STANDARD - full pipeline
        return TaskClassification(
            complexity=TaskComplexity.STANDARD,
            confidence=0.6,
            reasoning=f"Standard task: {word_count} words, simple={simple_matches}, complex={complex_matches}, files={has_files}",
            recommended_max_tokens=4096,
            use_repomix=True,
            skip_round2=False,
            skip_red_team=False
        )


def create_classifier() -> TaskClassifier:
    """Factory function to create task classifier."""
    return TaskClassifier()


def classify_task(task_description: str, relevant_files: Optional[list] = None) -> TaskClassification:
    """Convenience function to classify a task."""
    return create_classifier().classify(task_description, relevant_files)
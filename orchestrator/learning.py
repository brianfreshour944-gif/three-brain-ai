"""Adaptive learning system for three-brain AI - learns from feedback and optimizes pipeline."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

from orchestrator.router import TaskComplexity, TaskClassification
from orchestrator.task_store import Task, get_task_store
from orchestrator.memory import get_memory_manager
from orchestrator.config import MINISTRAL_MAX_TOKENS, DEEPSEEK_MAX_TOKENS, STRATEGIST_MAX_TOKENS

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Type of human feedback."""
    APPROVE = "approve"
    REJECT = "reject"
    REVISION = "revision"
    AUTO_APPROVE = "auto_approve"


class PipelineStage(Enum):
    """Pipeline stages that can be optimized."""
    CLASSIFY = "classify"
    ROUND1_BUILDER = "round1_builder"
    ROUND1_ANALYST = "round1_analyst"
    ROUND1_STRATEGIST = "round1_strategist"
    ROUND2_BUILDER = "round2_builder"
    ROUND2_ANALYST = "round2_analyst"
    ROUND2_STRATEGIST = "round2_strategist"
    RED_TEAM = "red_team"
    APPROVAL = "approval"


@dataclass
class FeedbackRecord:
    """Record of human feedback on a task."""
    id: str
    task_id: str
    task_description: str
    complexity: TaskComplexity
    pipeline_config: Dict[str, Any]  # Which stages were used, tokens, etc.
    feedback_type: FeedbackType
    feedback_notes: str
    timestamp: str
    pipeline_duration_ms: int
    context_tokens: int
    success: bool  # True if approved, False if rejected/revision


@dataclass
class ComplexityProfile:
    """Learned profile for a task complexity class."""
    complexity: TaskComplexity
    total_tasks: int = 0
    approved: int = 0
    rejected: int = 0
    revised: int = 0
    avg_duration_ms: float = 0.0
    avg_context_tokens: float = 0.0
    preferred_pipeline: str = "standard"  # trivial, simple, standard, complex
    optimal_reserve_tokens: int = 2000
    optimal_use_repomix: bool = True
    skip_red_team_threshold: float = 0.8  # Approval rate threshold
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PipelineOptimization:
    """Optimized pipeline configuration for a complexity class."""
    complexity: TaskComplexity
    use_repomix: bool
    reserve_tokens: int
    skip_round2: bool
    skip_red_team: bool
    builder_max_tokens: int
    analyst_max_tokens: int
    strategist_max_tokens: int
    confidence: float  # 0-1, based on sample size
    updated_at: str


class PreferenceLearner:
    """Learns preferences from human feedback."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
        self._load_profiles()
    
    def _init_db(self):
        """Initialize SQLite database for feedback storage."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_description TEXT NOT NULL,
                    complexity TEXT NOT NULL,
                    pipeline_config TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    feedback_notes TEXT,
                    timestamp TEXT NOT NULL,
                    pipeline_duration_ms INTEGER,
                    context_tokens INTEGER,
                    success INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS complexity_profiles (
                    complexity TEXT PRIMARY KEY,
                    total_tasks INTEGER DEFAULT 0,
                    approved INTEGER DEFAULT 0,
                    rejected INTEGER DEFAULT 0,
                    revised INTEGER DEFAULT 0,
                    avg_duration_ms REAL DEFAULT 0,
                    avg_context_tokens REAL DEFAULT 0,
                    preferred_pipeline TEXT DEFAULT 'standard',
                    optimal_reserve_tokens INTEGER DEFAULT 2000,
                    optimal_use_repomix INTEGER DEFAULT 1,
                    skip_red_team_threshold REAL DEFAULT 0.8,
                    last_updated TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_optimizations (
                    complexity TEXT PRIMARY KEY,
                    use_repomix INTEGER,
                    reserve_tokens INTEGER,
                    skip_round2 INTEGER,
                    skip_red_team INTEGER,
                    builder_max_tokens INTEGER,
                    analyst_max_tokens INTEGER,
                    strategist_max_tokens INTEGER,
                    confidence REAL,
                    updated_at TEXT
                )
            """)
            conn.commit()
    
    def _load_profiles(self):
        """Load complexity profiles from database."""
        self.profiles: Dict[TaskComplexity, ComplexityProfile] = {}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM complexity_profiles")
            for row in cursor.fetchall():
                profile = ComplexityProfile(
                    complexity=TaskComplexity(row["complexity"]),
                    total_tasks=row["total_tasks"],
                    approved=row["approved"],
                    rejected=row["rejected"],
                    revised=row["revised"],
                    avg_duration_ms=row["avg_duration_ms"],
                    avg_context_tokens=row["avg_context_tokens"],
                    preferred_pipeline=row["preferred_pipeline"],
                    optimal_reserve_tokens=row["optimal_reserve_tokens"],
                    optimal_use_repomix=bool(row["optimal_use_repomix"]),
                    skip_red_team_threshold=row["skip_red_team_threshold"],
                    last_updated=row["last_updated"],
                )
                self.profiles[profile.complexity] = profile
        
        # Ensure all complexities have profiles
        for complexity in TaskComplexity:
            if complexity not in self.profiles:
                self.profiles[complexity] = ComplexityProfile(complexity=complexity)
    
    def record_feedback(self, feedback: FeedbackRecord) -> None:
        """Record human feedback and update profiles."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO feedback 
                    (id, task_id, task_description, complexity, pipeline_config, 
                     feedback_type, feedback_notes, timestamp, pipeline_duration_ms,
                     context_tokens, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.id, feedback.task_id, feedback.task_description,
                    feedback.complexity.value, json.dumps(feedback.pipeline_config),
                    feedback.feedback_type.value, feedback.feedback_notes,
                    feedback.timestamp, feedback.pipeline_duration_ms,
                    feedback.context_tokens, int(feedback.success)
                ))
                conn.commit()
            
            # Update complexity profile
            self._update_profile(feedback)
            self._recompute_optimizations()
    
    def _update_profile(self, feedback: FeedbackRecord) -> None:
        """Update complexity profile based on feedback."""
        profile = self.profiles[feedback.complexity]
        profile.total_tasks += 1
        
        if feedback.feedback_type == FeedbackType.APPROVE:
            profile.approved += 1
        elif feedback.feedback_type == FeedbackType.REJECT:
            profile.rejected += 1
        elif feedback.feedback_type == FeedbackType.REVISION:
            profile.revised += 1
        
        # Update running averages
        n = profile.total_tasks
        profile.avg_duration_ms = ((profile.avg_duration_ms * (n - 1)) + feedback.pipeline_duration_ms) / n
        profile.avg_context_tokens = ((profile.avg_context_tokens * (n - 1)) + feedback.context_tokens) / n
        
        # Update preferred pipeline based on success rate
        if n >= 5:
            approval_rate = profile.approved / n
            if approval_rate > 0.8 and profile.preferred_pipeline != "trivial":
                # If very high approval, maybe we can simplify
                pass
            elif approval_rate < 0.5 and profile.preferred_pipeline != "complex":
                # If low approval, need more rigorous pipeline
                pass
        
        profile.last_updated = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO complexity_profiles
                (complexity, total_tasks, approved, rejected, revised,
                 avg_duration_ms, avg_context_tokens, preferred_pipeline,
                 optimal_reserve_tokens, optimal_use_repomix,
                 skip_red_team_threshold, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.complexity.value, profile.total_tasks, profile.approved,
                profile.rejected, profile.revised, profile.avg_duration_ms,
                profile.avg_context_tokens, profile.preferred_pipeline,
                profile.optimal_reserve_tokens, int(profile.optimal_use_repomix),
                profile.skip_red_team_threshold, profile.last_updated
            ))
            conn.commit()
    
    def _recompute_optimizations(self) -> None:
        """Recompute optimal pipeline configurations based on learned profiles."""
        with sqlite3.connect(self.db_path) as conn:
            for complexity in TaskComplexity:
                profile = self.profiles[complexity]
                n = profile.total_tasks
                
                if n < 3:
                    confidence = 0.1
                elif n < 10:
                    confidence = 0.5
                elif n < 30:
                    confidence = 0.75
                else:
                    confidence = 0.9
                
                # Determine optimal settings based on learned patterns
                approval_rate = profile.approved / max(n, 1)
                
                # Adjust repomix usage based on task complexity and approval rate
                if complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE):
                    use_repomix = False
                    reserve_tokens = 1000
                    skip_round2 = True
                    skip_red_team = True
                elif complexity == TaskComplexity.STANDARD:
                    use_repomix = True
                    reserve_tokens = 3000
                    skip_round2 = False
                    skip_red_team = approval_rate < 0.6
                else:  # COMPLEX
                    use_repomix = True
                    reserve_tokens = 5000
                    skip_round2 = False
                    skip_red_team = False
                
                # Adjust max tokens based on context usage
                avg_tokens = profile.avg_context_tokens
                if avg_tokens > 15000:
                    builder_tokens = 2048
                    analyst_tokens = 2048
                    strategist_tokens = 8192
                elif avg_tokens > 8000:
                    builder_tokens = 1536
                    analyst_tokens = 1536
                    strategist_tokens = 4096
                else:
                    builder_tokens = 1024
                    analyst_tokens = 1024
                    strategist_tokens = 2048
                
                optimization = PipelineOptimization(
                    complexity=complexity,
                    use_repomix=use_repomix,
                    reserve_tokens=reserve_tokens,
                    skip_round2=skip_round2,
                    skip_red_team=skip_red_team,
                    builder_max_tokens=builder_tokens,
                    analyst_max_tokens=analyst_tokens,
                    strategist_max_tokens=strategist_tokens,
                    confidence=confidence,
                    updated_at=datetime.now().isoformat(),
                )
                
                conn.execute("""
                    INSERT OR REPLACE INTO pipeline_optimizations
                    (complexity, use_repomix, reserve_tokens, skip_round2, skip_red_team,
                     builder_max_tokens, analyst_max_tokens, strategist_max_tokens,
                     confidence, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    complexity.value, int(use_repomix), reserve_tokens,
                    int(skip_round2), int(skip_red_team),
                    builder_tokens, analyst_tokens, strategist_tokens,
                    confidence, optimization.updated_at
                ))
            conn.commit()
    
    def get_optimization(self, complexity: TaskComplexity) -> Optional[PipelineOptimization]:
        """Get optimized pipeline configuration for a complexity class."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM pipeline_optimizations WHERE complexity = ?",
                (complexity.value,)
            )
            row = cursor.fetchone()
            if row:
                return PipelineOptimization(
                    complexity=TaskComplexity(row["complexity"]),
                    use_repomix=bool(row["use_repomix"]),
                    reserve_tokens=row["reserve_tokens"],
                    skip_round2=bool(row["skip_round2"]),
                    skip_red_team=bool(row["skip_red_team"]),
                    builder_max_tokens=row["builder_max_tokens"],
                    analyst_max_tokens=row["analyst_max_tokens"],
                    strategist_max_tokens=row["strategist_max_tokens"],
                    confidence=row["confidence"],
                    updated_at=row["updated_at"],
                )
        return None
    
    def get_profile(self, complexity: TaskComplexity) -> ComplexityProfile:
        return self.profiles.get(complexity, ComplexityProfile(complexity=complexity))
    
    def get_approval_rate(self, complexity: TaskComplexity) -> float:
        profile = self.profiles.get(complexity)
        if not profile or profile.total_tasks == 0:
            return 0.0
        return profile.approved / profile.total_tasks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT COUNT(*) as total, SUM(success) as approved FROM feedback")
            row = cursor.fetchone()
            total = row["total"] or 0
            approved = row["approved"] or 0
        
        return {
            "total_feedback": total,
            "approved": approved,
            "rejected": total - approved,
            "approval_rate": approved / total if total > 0 else 0,
            "profiles": {
                c.value: {
                    "total": p.total_tasks,
                    "approved": p.approved,
                    "approval_rate": p.approved / max(p.total_tasks, 1),
                    "avg_duration_ms": p.avg_duration_ms,
                    "avg_context_tokens": p.avg_context_tokens,
                }
                for c, p in self.profiles.items()
            }
        }


class AdaptivePipeline:
    """Pipeline that adapts based on learned preferences."""
    
    def __init__(self, learner: PreferenceLearner):
        self.learner = learner
    
    def get_pipeline_config(self, classification: TaskClassification) -> Dict[str, Any]:
        """Get optimized pipeline configuration for a task classification."""
        optimization = self.learner.get_optimization(classification.complexity)
        profile = self.learner.get_profile(classification.complexity)
        
        # Base configuration from classification
        config = {
            "complexity": classification.complexity.value,
            "use_repomix": classification.use_repomix,
            "reserve_tokens": classification.recommended_max_tokens,
            "skip_round2": classification.skip_round2,
            "skip_red_team": classification.skip_red_team,
            "max_tokens": {
                "builder": MINISTRAL_MAX_TOKENS,
                "analyst": DEEPSEEK_MAX_TOKENS,
                "strategist": STRATEGIST_MAX_TOKENS,
            },
            "reserve_tokens": classification.recommended_max_tokens,
            "use_repomix": classification.use_repomix,
        }
        
        # Apply learned optimizations if available and confident
        if classification.complexity in (TaskComplexity.STANDARD, TaskComplexity.COMPLEX):
            # Could adjust based on learned patterns
            pass
        
        return config
    
    def record_outcome(self, classification: TaskClassification, feedback_type: FeedbackType,
                       pipeline_config: Dict, duration_ms: int, context_tokens: int,
                       notes: str = "") -> None:
        """Record the outcome of a task execution."""
        feedback = FeedbackRecord(
            id=str(uuid4())[:8],
            task_id="",  # Will be filled by caller
            task_description="",  # Will be filled by caller
            complexity=classification.complexity,
            pipeline_config=pipeline_config,
            feedback_type=feedback_type,
            feedback_notes=notes,
            timestamp=datetime.now().isoformat(),
            pipeline_duration_ms=duration_ms,
            context_tokens=context_tokens,
            success=feedback_type in (FeedbackType.APPROVE, FeedbackType.AUTO_APPROVE),
        )
        self.learner.record_feedback(feedback)


def create_preference_learner(db_path: Optional[Path] = None) -> PreferenceLearner:
    """Factory to create preference learner."""
    if db_path is None:
        db_path = Path.home() / ".three-brain-ai" / "learning.db"
    return PreferenceLearner(db_path)


def create_adaptive_pipeline(learner: Optional[PreferenceLearner] = None) -> AdaptivePipeline:
    """Factory to create adaptive pipeline."""
    if learner is None:
        learner = create_preference_learner()
    return AdaptivePipeline(learner)
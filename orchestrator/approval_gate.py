"""Approval gate for three-brain AI system - handles human approval workflow."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4

from orchestrator.task_store import Task, get_task_store
from orchestrator.memory import get_memory_manager

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISIONS_REQUESTED = "revisions_requested"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """Request for human approval."""
    id: str
    task_id: str
    task_description: str
    proposed_plan: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None
    notes: str = ""
    reviewer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ApprovalGate:
    """Manages human approval workflow for task execution."""
    
    def __init__(
        self,
        project_root: Path,
        approval_callback: Optional[Callable[[ApprovalRequest], Awaitable[ApprovalStatus]]] = None,
        auto_approve_trivial: bool = True,
        auto_reject_on_timeout: bool = False,
        timeout_hours: int = 24,
    ):
        self.project_root = Path(project_root)
        self.approval_callback = approval_callback
        self.auto_approve_trivial = auto_approve_trivial
        self.auto_reject_on_timeout = auto_reject_on_timeout
        self.timeout_hours = timeout_hours
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.task_store = get_task_store()
        self.memory = get_memory_manager()
        
        # Approval storage directory
        self.approval_dir = self.project_root / "approvals"
        self.approval_dir.mkdir(parents=True, exist_ok=True)
    
    def create_approval_request(
        self,
        task_id: str,
        task_description: str,
        proposed_plan: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """Create a new approval request."""
        request = ApprovalRequest(
            id=str(uuid4())[:8],
            task_id=task_id,
            task_description=task_description,
            proposed_plan=proposed_plan,
            metadata=metadata or {},
        )
        
        self.pending_approvals[request.id] = request
        self._save_approval(request)
        logger.info(f"Created approval request {request.id} for task {task_id}")
        return request
    
    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Request approval - either via callback or auto-approve."""
        if self.auto_approve_trivial:
            # Check if task is trivial (simple, low risk)
            if self._is_trivial_request(request):
                logger.info(f"Auto-approving trivial request {request.id}")
                request.status = ApprovalStatus.AUTO_APPROVED
                request.decided_at = datetime.now().isoformat()
                request.decided_by = "auto"
                self._save_approval(request)
                return ApprovalStatus.AUTO_APPROVED
        
        # Check for existing decision
        if request.status != ApprovalStatus.PENDING:
            return request.status
        
        # Use callback if available
        if self.approval_callback:
            logger.info(f"Requesting approval via callback for {request.id}")
            status = await self.approval_callback(request)
            request.status = status
            request.decided_at = datetime.now().isoformat()
            self._save_approval(request)
            return status
        
        # Fallback: create approval file for manual review
        logger.info(f"Creating approval file for manual review: {request.id}")
        self._create_approval_file(request)
        
        # Wait for decision (polling)
        return await self._wait_for_decision(request)
    
    async def _wait_for_decision(self, request: ApprovalRequest, poll_interval: int = 5) -> ApprovalStatus:
        """Wait for human decision via approval file."""
        approval_file = self.approval_dir / f"{request.id}.approval"
        
        while request.status == ApprovalStatus.PENDING:
            await asyncio.sleep(poll_interval)
            
            # Check if approval file was created
            if approval_file.exists():
                try:
                    content = approval_file.read_text().strip()
                    if content.lower() in ("approve", "approved", "yes", "y"):
                        request.status = ApprovalStatus.APPROVED
                        request.decided_by = "human"
                    elif content.lower() in ("reject", "rejected", "no", "n"):
                        request.status = ApprovalStatus.REJECTED
                        request.decided_by = "human"
                    else:
                        continue  # Invalid content, keep waiting
                    
                    request.decided_at = datetime.now().isoformat()
                    self._save_approval(request)
                    approval_file.unlink()  # Remove approval file
                    return request.status
                except Exception:
                    continue
            
            # Check timeout
            created = datetime.fromisoformat(request.created_at)
            if (datetime.now() - created).total_seconds() > self.timeout_hours * 3600:
                if self.auto_reject_on_timeout:
                    request.status = ApprovalStatus.EXPIRED
                    request.decided_by = "timeout"
                    request.decided_at = datetime.now().isoformat()
                    self._save_approval(request)
                    return ApprovalStatus.EXPIRED
        
        return request.status
    
    def _is_trivial_request(self, request: ApprovalRequest) -> bool:
        """Check if request is trivial (auto-approvable)."""
        # Simple heuristics for trivial tasks
        desc = request.task_description.lower()
        plan = request.proposed_plan.lower()
        
        # Very short descriptions
        if len(request.task_description.split()) <= 5:
            return True
        
        # Simple operations
        trivial_keywords = {"add", "fix", "change", "update", "rename", "remove", "delete"}
        if any(kw in desc for kw in trivial_keywords) and len(desc.split()) < 15:
            return True
        
        return False
    
    def _save_approval(self, request: ApprovalRequest) -> None:
        """Save approval request to file."""
        request.updated_at = datetime.now().isoformat()
        approval_file = self.approval_dir / f"{request.id}.json"
        import json
        data = {
            "id": request.id,
            "task_id": request.task_id,
            "task_description": request.task_description,
            "proposed_plan": request.proposed_plan,
            "status": request.status.value,
            "created_at": request.created_at,
            "updated_at": request.updated_at,
            "decided_at": request.decided_at,
            "decided_by": request.decided_by,
            "notes": request.notes,
            "reviewer": request.reviewer,
            "metadata": request.metadata,
        }
        approval_file.write_text(json.dumps(data, indent=2))
    
    def _create_approval_file(self, request: ApprovalRequest) -> None:
        """Create approval file for manual review."""
        approval_file = self.approval_dir / f"{request.id}.approval"
        content = f"""# Approval Request: {request.id}
Task: {request.task_description}
Plan: {request.proposed_plan}

To approve: echo "approve" > {request.id}.approval
To reject: echo "reject" > {request.id}.approval
"""
        approval_file.write_text(content)
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [r for r in self.pending_approvals.values() if r.status == ApprovalStatus.PENDING]
    
    def get_approval(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Get approval request by ID."""
        return self.pending_approvals.get(approval_id)
    
    def approve(self, approval_id: str, notes: str = "", reviewer: str = "") -> bool:
        """Approve a pending request."""
        request = self.pending_approvals.get(approval_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return False
        
        request.status = ApprovalStatus.APPROVED
        request.decided_at = datetime.now().isoformat()
        request.decided_by = reviewer or "human"
        request.notes = notes
        self._save_approval(request)
        
        # Record in memory
        self.memory.add_decision(
            decision=f"Approved task {request.task_id}",
            rationale=notes or "Human approved",
            context=request.task_description,
        )
        return True
    
    def reject(self, approval_id: str, notes: str = "", reviewer: str = "") -> bool:
        """Reject a pending request."""
        request = self.pending_approvals.get(approval_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return False
        
        request.status = ApprovalStatus.REJECTED
        request.decided_at = datetime.now().isoformat()
        request.decided_by = reviewer or "human"
        request.notes = notes
        self._save_approval(request)
        
        # Record in memory
        self.memory.add_decision(
            decision=f"Rejected task {request.task_id}",
            rationale=notes,
            context=request.task_description,
        )
        return True
    
    def request_revision(self, approval_id: str, notes: str = "", reviewer: str = "") -> bool:
        """Request revisions to a pending request."""
        request = self.pending_approvals.get(approval_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return False
        
        request.status = ApprovalStatus.REVISIONS_REQUESTED
        request.decided_at = datetime.now().isoformat()
        request.decided_by = reviewer or "human"
        request.notes = notes
        self._save_approval(request)
        return True
    
    def cleanup_expired(self) -> int:
        """Clean up expired approval requests."""
        cleaned = 0
        now = datetime.now()
        for request in list(self.pending_approvals.values()):
            if request.status == ApprovalStatus.PENDING:
                created = datetime.fromisoformat(request.created_at)
                if (now - created).total_seconds() > self.timeout_hours * 3600:
                    request.status = ApprovalStatus.EXPIRED
                    request.decided_at = datetime.now().isoformat()
                    request.decided_by = "timeout"
                    self._save_approval(request)
                    cleaned += 1
        return cleaned


# Convenience functions
def create_approval_gate(project_root: Path, **kwargs) -> ApprovalGate:
    """Factory function to create approval gate."""
    return ApprovalGate(project_root=Path(project_root), **kwargs)


def quick_approve(task_id: str, plan: str, project_root: Path) -> bool:
    """Quick approval for trivial tasks."""
    gate = create_approval_gate(project_root, auto_approve_trivial=True)
    request = gate.create_approval_request(
        task_id=task_id,
        task_description="Trivial task",
        proposed_plan=plan,
    )
    # Auto-approves trivial tasks
    asyncio.run(gate.request_approval(gate.pending_approvals[list(gate.pending_approvals.keys())[0]]))
    return True
"""Scope checking for safety system - prevents modification of protected files/directories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Default protected paths - files/directories that should not be modified without approval
DEFAULT_PROTECTED_PATHS = {
    # Environment & secrets
    ".env", ".env.*", "*.key", "*.pem", "*.p12", "*.pfx", "*.crt", "*.cer",
    "credentials/", "secrets/", ".secrets/", "vault/", ".vault/",
    
    # Critical infrastructure
    "docker-compose*.yml", "docker-compose*.yaml", "Dockerfile*", "dockerfile*",
    "kubernetes/", "k8s/", "helm/", "*.tf", "*.tfvars", "terraform/",
    
    # Deployment & CI/CD
    ".github/workflows/", ".gitlab-ci.yml", ".circleci/", "Jenkinsfile",
    ".travis.yml", "azure-pipelines.yml", "bitbucket-pipelines.yml",
    
    # Core application
    "risk_engine.py", "position_sizing.py", "order_execution.py", "broker.py",
    "trading/", "strategy/", "portfolio/", "execution/",
    
    # Configuration
    "config.yaml", "config.yml", "config.json", "settings.py", "settings.yaml",
    "constants.py", "constants.yaml",
    
    # Database & migrations
    "migrations/", "alembic/", "*.sql", "schema.*",
    
    # Security
    "security/", "auth/", "authentication/", "authorization/",
    "middleware/security.py", "middleware/auth.py",
    
    # Infrastructure
    "infrastructure/", "infra/", "scripts/deploy*", "scripts/migrate*",
    "scripts/backup*", "scripts/restore*",
}

# Default protected file patterns (globs)
DEFAULT_PROTECTED_PATTERNS = [
    "*.env", "*.env.*", ".env", ".env.*",
    "*.key", "*.pem", "*.p12", "*.pfx", "*.crt", "*.cer",
    "*.tf", "*.tfvars",
    "*.tfstate", "*.tfstate.backup",
    "id_rsa", "id_ed25519", "id_dsa", "*.ppk",
    "*.kubeconfig", "kubeconfig",
    "credentials.json", "service-account*.json",
    "*.tfstate", "*.tfstate.*",
    "secrets.yaml", "secrets.yml",
    "vault.yaml", "vault.yml",
]


class ScopeChecker:
    """Checks if file modifications are within allowed scope."""
    
    def __init__(
        self,
        project_root: Path,
        protected_paths: Optional[Set[str]] = None,
        protected_patterns: Optional[Set[str]] = None,
        allowed_paths: Optional[Set[str]] = None,
    ):
        self.project_root = project_root.resolve()
        self.protected_paths = protected_paths or DEFAULT_PROTECTED_PATHS
        self.protected_patterns = protected_patterns or set(DEFAULT_PROTECTED_PATTERNS)
        self.allowed_paths = allowed_paths or set()
    
    def _matches_pattern(self, path: Path, patterns: Set[str]) -> bool:
        """Check if path matches any pattern."""
        rel_path = path.relative_to(self.project_root) if path.is_absolute() else path
        path_str = str(rel_path)
        
        for pattern in patterns:
            if self._match_glob(pattern, path_str):
                return True
        return False
    
    def _match_glob(self, pattern: str, path: str) -> bool:
        """Simple glob matching."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)
    
    def is_protected(self, file_path: Path) -> bool:
        """Check if a file is protected."""
        try:
            if not file_path.exists():
                # Check parent directories
                for parent in file_path.parents:
                    if self._matches_pattern(parent, self.protected_paths):
                        return True
                return self._matches_pattern(file_path, self.protected_patterns)
            
            if self._matches_pattern(file_path, self.protected_paths):
                return True
            
            return self._matches_pattern(file_path, self.protected_patterns)
        except Exception:
            return False
    
    def is_allowed(self, file_path: Path) -> bool:
        """Check if a file is explicitly allowed."""
        if not self.allowed_paths:
            return False
        try:
            for allowed in self.allowed_paths:
                if file_path.match(allowed) or any(
                    p.match(allowed) for p in file_path.parents
                ):
                    return True
        except Exception:
            pass
        return False
    
    def check_modification(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """Check if modification is allowed.
        
        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        # If explicitly allowed, permit
        if self.is_allowed(file_path):
            return True, None
        
        # If protected and not explicitly allowed, block
        if self.is_protected(file_path):
            return False, f"File '{file_path}' is protected and requires approval"
        
        return True, None


def check_scope(file_path: Path, project_root: Optional[Path] = None) -> tuple[bool, List[str]]:
    """Convenience function to check if a file modification is allowed."""
    root = project_root or Path.cwd()
    checker = ScopeChecker(project_root=Path(project_root) if project_root else Path.cwd())
    allowed, reason = checker.check_modification(file_path)
    if not allowed:
        return False, [reason] if reason else ["File is protected"]
    return True, []


def check_directory_scope(directory: Path, project_root: Optional[Path] = None) -> tuple[bool, List[str]]:
    """Check all files in a directory for scope violations."""
    root = project_root or Path.cwd()
    checker = ScopeChecker(project_root=Path(project_root) if project_root else Path.cwd())
    
    all_errors = []
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            allowed, reason = checker.check_modification(file_path)
            if not allowed:
                all_errors.append(f"{file_path}: {reason}")
    
    return len(all_errors) == 0, all_errors
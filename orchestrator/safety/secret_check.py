"""Secret detection for safety system."""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Common secret patterns
SECRET_PATTERNS = [
    # API Keys
    (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})[\"']?", "API Key"),
    (r"(?i)(secret[_-]?key|secretkey)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})[\"']?", "Secret Key"),
    (r"(?i)(access[_-]?token|accesstoken)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})[\"']?", "Access Token"),
    (r"(?i)(auth[_-]?token|authtoken)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})[\"']?", "Auth Token"),
    (r"(?i)(bearer[_-]?token|bearertoken)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})[\"']?", "Bearer Token"),
    
    # Cloud Provider Keys
    (r"(?i)(aws[_-]?access[_-]?key|awsaccesskey)\s*[:=]\s*[\"']?([A-Z0-9]{20})[\"']?", "AWS Access Key"),
    (r"(?i)(aws[_-]?secret[_-]?key|awssecretkey)\s*[:=]\s*[\"']?([a-zA-Z0-9/+=]{40})[\"']?", "AWS Secret Key"),
    (r"(?i)(gcp[_-]?key|gcpkey)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})[\"']?", "GCP Key"),
    (r"(?i)(azure[_-]?key|azurekey)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})[\"']?", "Azure Key"),
    
    # Database
    (r"(?i)(db[_-]?password|dbpassword|database[_-]?password)\s*[:=]\s*[\"']?([^\s\"']{8,})[\"']?", "Database Password"),
    (r"(?i)(postgres://|mysql://|mongodb://|redis://)[^:\s]+:[^@\s]+@", "Database Connection String"),
    
    # Private Keys
    (r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----", "Private Key"),
    (r"-----BEGIN (PGP|OPENPGP) PRIVATE KEY BLOCK-----", "PGP Private Key"),
    
    # JWT
    (r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "JWT Token"),
    
    # Generic high-entropy strings
    (r"[a-zA-Z0-9_\-]{32,}", "High-entropy string (possible secret)"),
]


def scan_secrets(content: str, file_path: str = "") -> List[Tuple[str, str, int]]:
    """Scan content for secrets.
    
    Returns:
        List of (pattern_name, matched_text, line_number)
    """
    findings = []
    lines = content.split("\n")
    
    for line_num, line in enumerate(lines, 1):
        for pattern, name in SECRET_PATTERNS:
            matches = re.finditer(pattern, line)
            for match in matches:
                # Skip if it looks like a placeholder
                matched = match.group(0)
                if any(placeholder in matched.lower() for placeholder in [
                    "your_", "your-", "example", "placeholder", "dummy", "test", 
                    "fake", "sample", "xxx", "yyy", "zzz", "change_me",
                    "insert_", "replace_", "add_", "put_"
                ]):
                    continue
                findings.append((name, matched[:100], line_num))
    
    return findings


def scan_file(file_path: Path) -> List[Tuple[str, str, int]]:
    """Scan a file for secrets."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return scan_secrets(content, str(file_path))
    except Exception as e:
        return [("ERROR", f"Failed to read file: {e}", 0)]


def scan_directory(directory: Path, extensions: Optional[list] = None) -> dict:
    """Scan a directory for secrets.
    
    Returns:
        Dict mapping file paths to list of findings
    """
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".json", ".yaml", ".yml", ".env", ".ini", ".cfg", ".conf", ".txt", ".md"]
    
    results = {}
    for ext in extensions:
        for file_path in directory.rglob(f"*{ext}"):
            if file_path.is_file():
                findings = scan_file(file_path)
                if findings:
                    results[str(file_path)] = findings
    
    return results


def check_secrets(file_path: Path) -> tuple[bool, List[str]]:
    """Check a single file for secrets.
    
    Returns:
        Tuple of (is_clean, list_of_findings)
    """
    findings = scan_file(file_path)
    if findings:
        errors = [f"Line {ln}: {name} - {matched}" for name, matched, ln in findings]
        return False, errors
    return True, []


def check_directory_secrets(directory: Path, extensions: Optional[list] = None) -> tuple[bool, List[str]]:
    """Check all files in a directory for secrets."""
    results = scan_directory(directory, extensions)
    all_errors = []
    for file_path, findings in results.items():
        for name, matched, line_num in findings:
            all_errors.append(f"{file_path}:{line_num}: {name} - {matched[:100]}")
    return len(all_errors) == 0, all_errors
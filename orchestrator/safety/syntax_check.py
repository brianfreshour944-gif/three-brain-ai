"""Syntax checking for safety system."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def check_syntax(file_path: Path) -> tuple[bool, List[str]]:
    """Check Python syntax of a file.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    try:
        content = file_path.read_text(encoding="utf-8")
        ast.parse(content)
        return True, []
    except SyntaxError as e:
        errors.append(f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}")
        return False, errors
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, errors


def check_directory_syntax(directory: Path, extensions: Optional[List[str]] = None) -> tuple[bool, List[str]]:
    """Check syntax of all Python files in a directory.
    
    Returns:
        Tuple of (all_valid, list_of_errors)
    """
    if extensions is None:
        extensions = [".py"]
    
    all_errors = []
    all_valid = True
    
    for ext in extensions:
        for file_path in directory.rglob(f"*{ext}"):
            if file_path.is_file():
                valid, errors = check_syntax(file_path)
                if not valid:
                    all_valid = False
                    for err in errors:
                        all_errors.append(f"{file_path}: {err}")
    
    return all_valid, all_errors
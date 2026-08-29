"""Import checking for safety system."""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to collect imports from Python code."""
    
    def __init__(self):
        self.imports: Set[str] = set()
        self.from_imports: Set[str] = set()
    
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.from_imports.add(node.module.split(".")[0])
        self.generic_visit(node)


def get_imports(file_path: Path) -> tuple[Set[str], Set[str]]:
    """Extract imports from a Python file.
    
    Returns:
        Tuple of (direct_imports, from_imports)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports, visitor.from_imports
    except Exception as e:
        logger.warning(f"Failed to parse imports from {file_path}: {e}")
        return set(), set()


def check_imports(file_path: Path, allowed_imports: Optional[Set[str]] = None,
                  blocked_imports: Optional[Set[str]] = None) -> tuple[bool, List[str]]:
    """Check if imports in a file are allowed.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    imports, from_imports = get_imports(file_path)
    all_imports = imports | from_imports
    
    if blocked_imports:
        blocked = all_imports & blocked_imports
        if blocked:
            for imp in blocked:
                errors.append(f"Blocked import: {imp}")
    
    if allowed_imports is not None:
        disallowed = all_imports - allowed_imports
        if disallowed:
            for imp in disallowed:
                errors.append(f"Disallowed import (not in allowlist): {imp}")
    
    return len(errors) == 0, errors


def check_directory_imports(directory: Path, allowed_imports: Optional[Set[str]] = None,
                           blocked_imports: Optional[Set[str]] = None,
                           extensions: Optional[List[str]] = None) -> tuple[bool, List[str]]:
    """Check imports in all Python files in a directory."""
    if extensions is None:
        extensions = [".py"]
    
    all_errors = []
    all_valid = True
    
    for ext in extensions:
        for file_path in directory.rglob(f"*{ext}"):
            if file_path.is_file():
                valid, errors = check_imports(file_path, allowed_imports, blocked_imports)
                if not valid:
                    all_valid = False
                    for err in errors:
                        all_errors.append(f"{file_path}: {err}")
    
    return all_valid, all_errors


# Default blocked imports for security
DEFAULT_BLOCKED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "pickle", "marshal",
    "ctypes", "cffi", "ctypes.util", "multiprocessing", "threading",
    "socket", "urllib", "requests", "http", "ftplib", "telnetlib",
    "smtplib", "poplib", "imaplib", "nntplib", "xmlrpc", "importlib",
    "pkgutil", "runpy", "code", "codeop", "builtins", "types",
    "inspect", "gc", "weakref", "copyreg", "shelve", "dbm",
    "sqlite3", "csv", "configparser", "netrc", "xdrlib", "plistlib",
    "hashlib", "hmac", "secrets", "uuid", "random", "statistics",
    "math", "decimal", "fractions", "itertools", "functools",
    "operator", "collections", "heapq", "bisect", "array",
    "struct", "string", "re", "difflib", "textwrap", "unicodedata",
    "stringprep", "readline", "rlcompleter", "argparse", "getopt",
    "optparse", "cmd", "shlex", "typing", "pydoc", "doctest",
    "unittest", "test", "test.support", "test.regrtest", "test.script_helper",
    "venv", "ensurepip", "zipapp", "py_compile", "compileall",
    "dis", "pickletools", "symbol", "opcode", "ast", "symtable",
    "tokenize", "token", "keyword", "parser", "parser", "tabnanny",
    "pyclbr", "py_compile", "compileall", "dis", "pickletools",
    "distutils", "setuptools", "pip", "wheel", "pkg_resources",
    "site", "usercustomize", "sitecustomize", "warnings", "traceback",
    "linecache", "codecs", "encodings", "locale", "gettext",
    "logging", "logging.config", "logging.handlers", "trace",
    "tracemalloc", "faulthandler", "cgitb", "pdb", "bdb", "faulthandler",
    "pprint", "reprlib", "numbers", "math", "cmath", "decimal",
    "fractions", "random", "statistics", "itertools", "functools",
    "operator", "collections", "heapq", "bisect", "array",
    "weakref", "types", "copy", "pprint", "reprlib", "enum",
    "graphlib", "contextlib", "abc", "atexit", "dataclasses",
    "inspect", "importlib", "importlib.abc", "importlib.machinery",
    "importlib.util", "importlib.metadata", "importlib.resources",
    "importlib.abc", "importlib.machinery", "importlib.util",
    "importlib.metadata", "importlib.resources", "importlib.abc",
    "importlib.machinery", "importlib.util", "importlib.metadata",
    "importlib.resources", "importlib.abc", "importlib.machinery",
    "importlib.util", "importlib.metadata", "importlib.resources"
}
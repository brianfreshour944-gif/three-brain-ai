"""Configuration for the three-brain system."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent.parent
TASKS_DIR = os.getenv("TASKS_DIR", str(BASE_DIR / "tasks"))
MEMORY_DIR = os.getenv("MEMORY_DIR", str(BASE_DIR / "memory"))
LOGS_DIR = os.getenv("LOGS_DIR", str(BASE_DIR / "logs"))

# Model endpoints
MINISTRAL_BASE_URL = os.getenv("MINISTRAL_BASE_URL", "http://localhost:8001/v1")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "http://localhost:8002/v1")

# Model names
MINISTRAL_MODEL = os.getenv("MINISTRAL_MODEL", "ministral")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek")
STRATEGIST_MODEL = os.getenv("STRATEGIST_MODEL", "anthropic/claude-3.5-sonnet")

# Temperatures
BUILDER_TEMP = float(os.getenv("BUILDER_TEMP", "0.3"))
ANALYST_TEMP = float(os.getenv("ANALYST_TEMP", "0.2"))
STRATEGIST_TEMP = float(os.getenv("STRATEGIST_TEMP", "0.1"))

# Context windows
MINISTRAL_CTX = int(os.getenv("MINISTRAL_CTX", "4096"))
DEEPSEEK_CTX = int(os.getenv("DEEPSEEK_CTX", "4096"))
STRATEGIST_CTX = int(os.getenv("STRATEGIST_CTX", "200000"))

# Max completion tokens (must fit within context window)
MINISTRAL_MAX_TOKENS = int(os.getenv("MINISTRAL_MAX_TOKENS", "2048"))
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048"))
STRATEGIST_MAX_TOKENS = int(os.getenv("STRATEGIST_MAX_TOKENS", "8192"))

# Safety
REQUIRE_APPROVAL = os.getenv("REQUIRE_APPROVAL", "true").lower() == "true"
BLOCKED_PATHS = [p.strip() for p in os.getenv("BLOCKED_PATHS", ".env,*.key,*.pem,secrets/,credentials/").split(",")]
PROTECTED_FILES = [p.strip() for p in os.getenv("PROTECTED_FILES", "risk_engine.py,position_sizing.py,order_execution.py,broker.py").split(",")]

# Ensure directories exist
for dir_path in [TASKS_DIR, MEMORY_DIR, LOGS_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
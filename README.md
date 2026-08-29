# Three-Brain AI Orchestrator

A local-first AI engineering assistant that uses three specialized LLMs to analyze, design, and implement solutions for existing projects.

## Architecture

| Agent | Role | Model | Endpoint |
|-------|------|-------|----------|
| **Builder** | Creates solutions, proposes architecture | Ministral (GGUF) | Local llama.cpp:8001 |
| **Analyst** | Finds flaws, critiques, suggests improvements | DeepSeek (GGUF) | Local llama.cpp:8002 |
| **Strategist** | Final architecture decisions, implementation plans | Claude 3.5 Sonnet | OpenRouter (cloud) |
| **Red Team** | Security review (optional) | Claude 3.5 Sonnet | OpenRouter (cloud) |

## Flow

```
User Task
    │
    ▼
┌─────────────────────┐
│ Context Manager     │ ← Project memory, relevant files
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
Builder     Analyst
(Parallel)  (Parallel)
    │           │
    └─────┬─────┘
          ▼
    Strategist
          │
          ▼
    Red Team (optional)
          │
          ▼
    Human Approval
          │
          ▼
┌─────────────────────┐
│ FINAL_IMPLEMENTATION_PLAN.md │ → JetBrains AI / opencode consumes this
└─────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd three-brain-ai
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### 2. Start Local LLM Servers

**Terminal 1 - Ministral (Builder):**
```bash
./llama-server \
  -m /path/to/models/ministral.gguf \
  -ngl 999 \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8001
```

**Terminal 2 - DeepSeek (Analyst):**
```bash
./llama-server \
  -m /path/to/models/deepseek.gguf \
  -ngl 999 \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8002
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY
```

### 4. Initialize in Your Project

```bash
cd /path/to/your/project
three-brain init
```

### 5. Run a Task

```bash
three-brain run "Add a retry mechanism with exponential backlog to the API client"
```

### 6. Review and Approve

```bash
# Check status
three-brain status

# Approve
three-brain approve <task_id> --notes "Looks good, implement it"

# Or reject
three-brain reject <task_id> --notes "Need different approach"

# Or request revisions
three-brain revise <task_id> --notes "Add more error handling"
```

## Commands

| Command | Description |
|---------|-------------|
| `three-brain run "task"` | Run task through pipeline |
| `three-brain status [task_id]` | Check task status |
| `three-brain approve <id>` | Approve task |
| `three-brain reject <id>` | Reject task |
| `three-brain revise <id>` | Request revisions |
| `three-brain memory` | View/manage project memory |
| `three-brain safety <file>` | Run safety checks |
| `three-brain health` | Check LLM endpoints |
| `three-brain init` | Initialize in project |

## Output

The system produces a structured `FINAL_IMPLEMENTATION_PLAN.md` that can be consumed by:
- **JetBrains AI Assistant** - Paste the plan, let it implement
- **opencode** - Feed the plan as context
- **Manual implementation** - Follow the step-by-step plan

## Safety Layer

Before any code execution:
- Syntax validation
- Import analysis (blocks dangerous modules)
- Path traversal prevention
- Secret detection
- Protected file guarding (requires explicit approval)
- Human approval gate (configurable)

## Project Memory

The system maintains persistent memory across sessions:
- User preferences
- Architecture decisions
- Lessons learned
- Decision log with rationale

## Requirements

- Python 3.10+
- llama.cpp server with CUDA (for local models)
- Ministral GGUF model (~7B params)
- DeepSeek GGUF model (~7B params)
- OpenRouter API key (for Strategist/Red Team)

## Directory Structure

```
three-brain-ai/
├── agents/
│   ├── __init__.py
│   └── llm_client.py
├── orchestrator/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── context_manager.py
│   ├── memory.py
│   ├── task_store.py
│   ├── router.py
│   └── safety.py
├── prompts/
│   ├── __init__.py
│   ├── builder.py
│   ├── analyst.py
│   ├── strategist.py
│   └── red_team.py
├── memory/           # Project memory (created at runtime)
├── tasks/            # Task storage (created at runtime)
├── logs/             # Logs (created at runtime)
├── cli.py
├── pyproject.toml
├── .env.example
└── README.md
```
# Three-Brain AI Orchestrator

A local-first AI engineering assistant that uses three specialized LLMs to analyze, design, and implement solutions for existing projects. When the primary Strategist model fails, it automatically falls back to a backup model and notifies you via Discord.

## Architecture

| Agent | Role | Model | Endpoint |
|-------|------|-------|----------|
| **Builder** | Creates solutions, proposes architecture | Ministral 7B (GGUF) | Kaggle GPU → FRP → Oracle :7001 |
| **Analyst** | Finds flaws, critiques, suggests improvements | DeepSeek 7B (GGUF) | Kaggle GPU → FRP → Oracle :7002 |
| **Strategist** | Final architecture decisions, implementation plans | Nemotron 3 Ultra (free) | OpenRouter (cloud) |
| **Strategist Fallback** | Backup when Nemotron fails | GLM-5.3 Flash | OpenRouter (cloud) |
| **Red Team** (optional) | Security review | Nemotron 3 Ultra (free) | OpenRouter (cloud) |

### Automatic Fallback

If the primary Strategist (Nemotron) fails (timeout, error, rate limit), the system automatically retries with GLM-5.3 Flash and sends a Discord notification:

- `:warning:` — Primary failed, fallback succeeded
- `:x:` — Both primary and fallback failed

## Infrastructure Overview

```
┌─────────────┐     Tailscale      ┌─────────────┐     FRP Tunnel      ┌─────────────┐
│   Your PC   │ ────────────────── │  Oracle VM  │ ────────────────── │   Kaggle    │
│  (RDP/SSH)  │   100.113.214.63   │ (always on) │   ports 7000-7002 │ (2A-T4 GPU) │
└─────────────┘                    └─────────────┘                    └─────────────┘
       │                                  │                                  │
       │ JetBrains Air                    │ Orchestrator API                  │ llama.cpp
       │ IDE                              │ Port 8005                         │ Ports 8001/8002
       └──────────────────────────────────┴──────────────────────────────────┘
   Implementation                     Planning                          Model Serving
```

### Why Two Tunnel Tools?

| Connection | Tool | Reason |
|------------|------|--------|
| PC → Oracle (RDP) | **Tailscale** | Works great; PC isn't sandboxed. Keeps RDP private. |
| Oracle → Kaggle (LLM) | **FRP** | Tailscale gets killed by Kaggle's sandbox (~5s). FRP survives. |

## Quick Start

### 1. Oracle VM Setup (One-time)

```bash
# Clone repo
git clone https://github.com/brianfreshour944-gif/three-brain-ai.git
cd three-brain-ai

# Configure environment
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY and DISCORD_WEBHOOK_URL

# Install FRP server (see oracle/frps.toml)
sudo cp oracle/frps.toml /home/ubuntu/frp_0.61.1_linux_arm64/frps.toml
sudo cp oracle/frps.service /etc/systemd/system/frps.service
sudo systemctl daemon-reload
sudo systemctl enable --now frps

# Open firewall ports (Oracle Cloud Console + iptables)
sudo iptables -I INPUT -p tcp --dport 7000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 7001 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 7002 -j ACCEPT
# Make iptables persistent:
sudo apt-get install -y iptables-persistent

# Start orchestrator
docker compose up -d
```

### 2. Kaggle Notebook (Every Session)

1. **Attach datasets** in Kaggle sidebar:
   - `brianfreshour/trading-bot-llms` (models)
   - `brianfreshour/trading-bot-wheels` (dependencies)

2. **Run the startup script** (copy from `kaggle/startup.py`):
   ```bash
   # In Kaggle notebook cell:
   !python3 /kaggle/working/startup.py
   ```

3. **Verify** from Oracle:
   ```bash
   curl http://127.0.0.1:7001/v1/models
   curl http://127.0.0.1:7002/v1/models
   ```

### 3. Run a Task

```bash
# From your project directory
three-brain run "Add retry mechanism with exponential backoff to the API client"

# Or via API
curl -X POST http://oracle:8005/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Add retry mechanism...", "auto_approve": false}'
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `three-brain run <task>` | Run a task through the pipeline |
| `three-brain approve <id>` | Approve a task for implementation |
| `three-brain reject <id> -n <reason>` | Reject a task |
| `three-brain revise <id> -n <notes>` | Request revisions |
| `three-brain status [id]` | Check task status |
| `three-brain health` | Check LLM endpoint health |
| `three-brain memory --show` | View project memory |
| `three-brain serve` | Start REST API server (port 8005) |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/tasks` | Submit a task |
| `GET` | `/health` | Check LLM health |
| `GET` | `/tasks/{id}` | Get task status |
| `POST` | `/tasks/{id}/approve` | Approve a task |
| `POST` | `/tasks/{id}/reject` | Reject a task |
| `POST` | `/tasks/{id}/revise` | Request revisions |

## Repository Structure

```
three-brain-ai/
├── README.md                    # This file
├── .env.example                 # Environment template
├── docker-compose.yml           # Orchestrator + dependencies
├── pyproject.toml               # Python package config
├── orchestrator/                # Core orchestration logic
│   ├── main.py                  # ThreeBrainOrchestrator
│   ├── config.py                # Configuration
│   ├── context_system.py        # Context building + Repomix caching
│   ├── memory.py                # Project memory (cached)
│   ├── router.py                # Task complexity classification
│   └── safety/                  # Safety checks (async)
├── agents/
│   ├── __init__.py              # Agent creation + auto-detection
│   └── llm_client.py            # LLM clients + FallbackLLMClient
├── prompts/                     # System prompts for each agent
│   ├── builder.py
│   ├── analyst.py
│   ├── strategist.py
│   └── red_team.py
├── kaggle/
│   └── startup.py               # Kaggle notebook startup script
├── oracle/
│   ├── frps.toml                # FRP server config
│   └── frps.service             # systemd service
└── three_brain_ai/
    └── cli.py                   # CLI + REST API
```

## Key Features

- **Automatic fallback** — Nemotron → GLM-5.3 Flash with Discord notifications
- **Smart routing** — 4 complexity levels (trivial/simple/standard/complex)
- **Two-round pipeline** — Round 1: parallel analysis → Round 2: sequential refinement
- **Connection pooling** — Shared HTTP connections to LLM endpoints
- **Repomix caching** — File-mtime-based cache avoids re-packing repo context
- **Async safety checks** — Non-blocking syntax/import/secret/scope validation
- **Auto-detection** — Automatically uses OpenRouter-only when Kaggle tunnel is down
- **Task enhancement** — Vague input → precise instructions via `enhance_task`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Required for Strategist/Red Team | - |
| `MINISTRAL_API_KEY` | Auth key for Builder endpoint | `dummy` |
| `DEEPSEEK_API_KEY` | Auth key for Analyst endpoint | `dummy` |
| `MINISTRAL_BASE_URL` | Builder endpoint | `http://127.0.0.1:7001/v1` |
| `DEEPSEEK_BASE_URL` | Analyst endpoint | `http://127.0.0.1:7002/v1` |
| `STRATEGIST_MODEL` | Primary Strategist model | `openrouter/nemotron-3-ultra-free` |
| `STRATEGIST_FALLBACK_MODEL` | Fallback Strategist model | `openrouter/z-ai/glm-5.3-flash` |
| `DISCORD_WEBHOOK_URL` | Discord webhook for notifications | - |
| `REQUIRE_APPROVAL` | Human approval gate | `true` |

## Integration

### opencode Plugin

The `orchestrator.js` plugin provides two tools:
- `orchestrator_task` — Delegate tasks to the orchestrator
- `enhance_task` — Convert vague input to precise instructions

### JetBrains / Hermes Commands

See `rpd-commands/` for ready-to-use commands:
- `jetbrains enhance` — Enhance vague task descriptions
- `hermes enhance` — Same for Hermes

## License

MIT

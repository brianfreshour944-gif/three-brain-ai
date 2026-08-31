# Three-Brain AI Orchestrator

A local-first AI engineering assistant that uses three specialized LLMs to analyze, design, and implement solutions for existing projects.

## Architecture

| Agent | Role | Model | Endpoint |
|-------|------|-------|----------|
| **Builder** | Creates solutions, proposes architecture | Ministral 7B (GGUF) | Kaggle GPU 0 → FRP → Oracle :7001 |
| **Analyst** | Finds flaws, critiques, suggests improvements | DeepSeek 7B (GGUF) | Kaggle GPU 1 → FRP → Oracle :7002 |
| **Strategist** | Final architecture decisions, implementation plans | Nemotron 3 Ultra (free) | OpenRouter (cloud) |
| **Red Team** (optional) | Security review | Nemotron 3 Ultra (free) | OpenRouter (cloud) |

## Infrastructure Overview

```
┌─────────────┐     Tailscale      ┌─────────────┐     FRP Tunnel      ┌─────────────┐
│   Your PC   │ ◄─────────────────► │  Oracle VM  │ ◄─────────────────► │  Kaggle     │
│  (RDP/SSH)  │   100.113.214.63   │  (always on)│   ports 7000-7002   │  (2×T4 GPU) │
└─────────────┘                     └─────────────┘                     └─────────────┘
       │                                    │                                    │
       │ JetBrains Air                      │ Orchestrator API                  │ llama.cpp
       │ IDE                                │ Port 8005                         │ Ports 8001/8002
       ▼                                    ▼                                    ▼
   Implementation                     Planning                          Model Serving
```

### Why Two Tunnel Tools?

| Connection | Tool | Reason |
|------------|------|--------|
| PC ↔ Oracle (RDP) | **Tailscale** | Works great; PC isn't sandboxed. Keeps RDP private. |
| Oracle ↔ Kaggle (LLM) | **FRP** | Tailscale gets killed by Kaggle's sandbox (~5s). FRP survives. |

## Quick Start

### 1. Oracle VM Setup (One-time)

```bash
# Clone repo
git clone https://github.com/brianfreshour944-gif/three-brain-ai.git
cd three-brain-ai

# Configure environment
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

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

## Repository Structure

```
three-brain-ai/
├── README.md                    # This file
├── .env.example                 # Orchestrator environment template
├── docker-compose.yml           # Orchestrator + dependencies (no LLMs)
├── pyproject.toml               # Python package config
├── orchestrator/                # Core orchestration logic
│   ├── main.py                  # ThreeBrainOrchestrator
│   ├── config.py                # Configuration
│   ├── context_system.py        # Context building + Repomix caching
│   ├── memory.py                # Project memory (cached)
│   ├── router.py                # Task complexity classification
│   ├── safety/                  # Safety checks (async)
│   └── ...
├── kaggle/
│   └── startup.py               # Kaggle notebook startup script
├── oracle/
│   ├── frps.toml                # FRP server config
│   └── frps.service             # systemd service
└── three_brain_ai/
    └── cli.py                   # CLI entry point
```

## Key Features

- **Connection pooling** - Shared HTTP/2 connections to LLM endpoints
- **Repomix caching** - File-mtime-based cache avoids re-packing repo context
- **Smart routing** - 4 complexity levels (trivial/simple/standard/complex) route to appropriate pipeline
- **Async safety checks** - Non-blocking syntax/import/secret/scope validation
- **Memory caching** - Project memory cached with mtime invalidation
- **Pre-compiled regex** - Router patterns compiled at module load

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Required for Strategist/Red Team | - |
| `MINISTRAL_BASE_URL` | Builder endpoint | `http://127.0.0.1:7001/v1` |
| `DEEPSEEK_BASE_URL` | Analyst endpoint | `http://127.0.0.1:7002/v1` |
| `STRATEGIST_MODEL` | OpenRouter model | `openrouter/nemotron-3-ultra-free` |
| `REQUIRE_APPROVAL` | Human approval gate | `true` |

## License

MIT
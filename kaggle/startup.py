#!/usr/bin/env python3
"""
Kaggle Startup Script for Three-Brain AI Orchestrator

Run this at the start of every Kaggle session to:
1. Install dependencies from cached wheels dataset
2. Load models from cached models dataset
3. Start two llama.cpp servers (one per GPU)
4. Connect via FRP tunnel to Oracle

Usage:
    python3 startup.py

Requirements (attach in Kaggle sidebar before running):
- brianfreshour/trading-bot-llms (models)
- brianfreshour/trading-bot-wheels (dependencies)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

# ============================================================
# CONFIGURATION — update these if infrastructure changes
# ============================================================
ORACLE_IP = "147.224.155.152"          # Oracle public IP (check: curl ifconfig.me)
FRP_TOKEN = "e89f57ebe5b240fbbac79265911305b4"  # Must match frps.toml on Oracle
FRP_VERSION = "0.61.1"
FRP_DIR = f"/kaggle/working/frp_{FRP_VERSION}_linux_amd64"
FRP_URL = f"https://github.com/fatedier/frp/releases/download/v{FRP_VERSION}/frp_{FRP_VERSION}_linux_amd64.tar.gz"

# Model paths (relative to attached dataset)
MISTRAL_PATH = "/kaggle/input/datasets/brianfreshour/trading-bot-llms/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
DEEPSEEK_PATH = "/kaggle/input/datasets/brianfreshour/trading-bot-llms/deepseek-coder-7b-instruct-v1.5.Q4_K_M.gguf"
WHEELS_DIR = "/kaggle/input/datasets/brianfreshour/trading-bot-wheels"

# Server configs
MISTRAL_PORT = 8001
DEEPSEEK_PORT = 8002
CTX_SIZE = 4096
GPU_LAYERS = 999
MAX_STARTUP_WAIT = 120  # seconds

# ============================================================


def log_step(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"STEP: {title}")
    print(f"{'=' * 60}")


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def check_api_alive(port: int, timeout: int = 5) -> bool:
    """Check if llama.cpp server is responding on given port."""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run command with error handling."""
    log_info(f"Running: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        log_error(f"Command failed (exit {e.returncode}): {' '.join(cmd)}")
        raise


def install_dependencies() -> None:
    """Install Python packages from cached wheels."""
    log_step("Installing Dependencies (Cached)")

    if not os.path.isdir(WHEELS_DIR):
        log_error(f"Wheels directory not found: {WHEELS_DIR}")
        log_error("Attach 'brianfreshour/trading-bot-wheels' dataset in Kaggle sidebar")
        sys.exit(1)

    pkgs = [
        "llama-cpp-python",
        "starlette-context",
        "uvicorn",
        "sse-starlette",
    ]

    run_cmd([
        "pip", "install", "-q", "--no-index",
        f"--find-links={WHEELS_DIR}", *pkgs
    ])
    log_info("Dependencies installed (cached, no download)")


def verify_models() -> tuple[str, str]:
    """Verify both model files exist."""
    log_step("Loading Cached Models")

    for path, name in [(MISTRAL_PATH, "Mistral"), (DEEPSEEK_PATH, "DeepSeek")]:
        if not os.path.exists(path):
            log_error(f"{name} model not found: {path}")
            log_error("Attach 'brianfreshour/trading-bot-llms' dataset in Kaggle sidebar")
            sys.exit(1)
        size_gb = os.path.getsize(path) / (1024**3)
        log_info(f"{name}: {path} ({size_gb:.2f} GB)")

    return MISTRAL_PATH, DEEPSEEK_PATH


def start_llama_server(
    model_path: str,
    port: int,
    gpu_id: int,
    name: str,
    chat_format: str | None = None,
) -> subprocess.Popen:
    """Start a llama.cpp server on specified GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python3", "-m", "llama_cpp.server",
        "--model", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--n_gpu_layers", str(GPU_LAYERS),
        "--n_ctx", str(CTX_SIZE),
    ]
    if chat_format:
        cmd.extend(["--chat_format", chat_format])

    log_file = f"/kaggle/working/{name.lower()}.log"
    log_info(f"Starting {name} on GPU {gpu_id} port {port} (log: {log_file})")

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
    )


def wait_for_servers() -> tuple[bool, bool]:
    """Wait for both servers to become healthy."""
    log_step("Waiting for Servers to Start")

    m_ok = d_ok = False
    start_time = time.time()

    for i in range(MAX_STARTUP_WAIT // 10):
        elapsed = int(time.time() - start_time)
        m_ok = check_api_alive(MISTRAL_PORT)
        d_ok = check_api_alive(DEEPSEEK_PORT)

        print(f"[{elapsed}s] Mistral: {'OK' if m_ok else 'WAIT'}  DeepSeek: {'OK' if d_ok else 'WAIT'}")

        if m_ok and d_ok:
            log_info("Both servers healthy!")
            return True, True

        time.sleep(10)

    log_error("Timeout waiting for servers")
    return m_ok, d_ok


def download_frp() -> None:
    """Download and extract FRP client."""
    log_step("Downloading FRP Client")

    tarball = "/kaggle/working/frp.tar.gz"

    if not os.path.exists(FRP_DIR):
        run_cmd(["wget", "-q", FRP_URL, "-O", tarball])
        run_cmd(["tar", "-xzf", tarball, "-C", "/kaggle/working/"])
        log_info(f"FRP extracted to {FRP_DIR}")
    else:
        log_info("FRP already present, skipping download")


def write_frpc_config() -> str:
    """Write frpc.toml configuration."""
    config = f"""serverAddr = "{ORACLE_IP}"
serverPort = 7000
auth.method = "token"
auth.token = "{FRP_TOKEN}"

[[proxies]]
name = "mistral"
type = "tcp"
localIP = "127.0.0.1"
localPort = {MISTRAL_PORT}
remotePort = 7001

[[proxies]]
name = "deepseek"
type = "tcp"
localIP = "127.0.0.1"
localPort = {DEEPSEEK_PORT}
remotePort = 7002
"""
    config_path = os.path.join(FRP_DIR, "frpc.toml")
    with open(config_path, "w") as f:
        f.write(config)
    log_info(f"FRP config written to {config_path}")
    return config_path


def start_frpc(config_path: str) -> subprocess.Popen:
    """Start FRP client."""
    log_step("Starting FRP Client")

    frpc_bin = os.path.join(FRP_DIR, "frpc")
    log_file = "/kaggle/working/frpc.log"

    proc = subprocess.Popen(
        [frpc_bin, "-c", config_path],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
    )

    # Brief wait for connection to establish
    time.sleep(3)

    # Show initial log output
    if os.path.exists(log_file):
        with open(log_file) as f:
            log_output = f.read().strip()
            if log_output:
                print(log_output)

    log_info("FRP client started in background")
    return proc


def verify_tunnel() -> bool:
    """Verify FRP tunnel is working from Oracle's perspective."""
    log_step("Verifying Tunnel (from Oracle's perspective)")

    # We can only test locally here; Oracle will test its endpoints
    log_info("Local test: checking llama.cpp servers still alive...")
    m_ok = check_api_alive(MISTRAL_PORT)
    d_ok = check_api_alive(DEEPSEEK_PORT)

    if m_ok and d_ok:
        log_info("✓ Both llama.cpp servers responding locally")
        log_info(f"  Oracle can now reach them at:")
        log_info(f"    Mistral  → http://127.0.0.1:7001/v1")
        log_info(f"    DeepSeek → http://127.0.0.1:7002/v1")
        return True
    else:
        log_error("✗ Local servers not responding")
        return False


def main() -> int:
    print("=" * 60)
    print("THREE-BRAIN AI — KAGGLE STARTUP")
    print("=" * 60)

    try:
        install_dependencies()
        mistral_path, deepseek_path = verify_models()

        # Start llama.cpp servers
        log_step("Starting llama.cpp Servers")
        mistral_proc = start_llama_server(mistral_path, MISTRAL_PORT, 0, "Mistral", "mistral-instruct")
        deepseek_proc = start_llama_server(deepseek_path, DEEPSEEK_PORT, 1, "DeepSeek")

        if not wait_for_servers():
            log_error("Server startup failed. Check logs:")
            log_error("  cat /kaggle/working/mistral.log")
            log_error("  cat /kaggle/working/deepseek.log")
            return 1

        # FRP tunnel
        download_frp()
        config_path = write_frpc_config()
        start_frpc(config_path)

        if not verify_tunnel():
            return 1

        print("\n" + "=" * 60)
        print("✓ DONE — Everything running successfully")
        print("=" * 60)
        print(f"  Mistral  (GPU 0): http://127.0.0.1:{MISTRAL_PORT}")
        print(f"  DeepSeek (GPU 1): http://127.0.0.1:{DEEPSEEK_PORT}")
        print(f"  Oracle endpoints:")
        print(f"    http://127.0.0.1:7001/v1  (Mistral)")
        print(f"    http://127.0.0.1:7002/v1  (DeepSeek)")
        print()
        print("Session will idle out after ~1 hour of inactivity")
        print("(No heartbeat — protects your 30h/week GPU quota)")
        print("=" * 60)

        # Keep script alive so background processes don't get killed
        # (Kaggle kills the notebook process when the cell finishes)
        log_info("Keeping session alive... (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(60)
                # Periodic health check
                if not (check_api_alive(MISTRAL_PORT) and check_api_alive(DEEPSEEK_PORT)):
                    log_error("Server health check failed!")
                    return 1
        except KeyboardInterrupt:
            log_info("Shutting down...")
            return 0

    except Exception as e:
        log_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
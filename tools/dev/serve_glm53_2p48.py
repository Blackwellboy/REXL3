#!/usr/bin/env python3
"""
Experimental dev launcher for GLM-5.3-Flash SAGE REXL3 2.48 bpw (single GB10).

- Does NOT modify sealed trees or approvals.
- Binds 127.0.0.1 on a separate port (default 8955); no TLS/auth; do not expose.
- No broad kills, no sudo needed; exits cleanly on Ctrl-C.
- Uses the engine's documented unvalidated-pack override for development:
  passes --allow-unvalidated-pack to rexl3-serve.

Usage:
  python3 tools/dev/serve_glm53_2p48.py \
    --serve-bin $HOME/rexl3/bin/rexl3-serve \
    --model-dir $HOME/models/glm53-2p48 \
    [--port 8955] [--max-model-len 131072] [--prefix-cache-mib 1024]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def require_file(p: Path, desc: str) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"Missing {desc}: {p}")


def require_executable(p: Path, desc: str) -> None:
    require_file(p, desc)
    if not os.access(p, os.X_OK):
        raise PermissionError(f"Not executable: {p}")


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) != 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--serve-bin", required=True, help="Path to native rexl3-serve binary")
    ap.add_argument("--model-dir", required=True, help="Path to vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw snapshot directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8955)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--prefix-cache-mib", type=int, default=1024)
    ap.add_argument("--served-model-name", default="GLM-5.3-Flash-SAGE-REXL3-2.48bpw")
    ap.add_argument("--reasoning-effort", default="low", choices=["low","medium","high","xhigh"])
    ap.add_argument("--log", default=str(Path.home()/".local/share/rexl3/dev-serve/log.txt"))
    ap.add_argument("--health-timeout-s", type=float, default=900.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    serve = Path(args.serve_bin).expanduser().resolve()
    model = Path(args.model_dir).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    require_executable(serve, "native server binary")
    if not model.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model}")
    for req in ("config.json", "generation_config.json", "tokenizer.json", "model.safetensors.index.json"):
        require_file(model/req, f"pack file {req}")

    # Print version (records actual runtime toolchain/driver fingerprint).
    ver = run([str(serve), "--version"], check=True)
    print("rexl3-serve --version:\n", ver.stdout.strip(), file=sys.stderr)

    # Optional read-only probes; ignore failures (not a production seal).
    try:
        fp = run([str(serve), "--print-fingerprints", "--model", str(model)], check=False)
        if fp.stdout.strip():
            print("pack fingerprints:\n", fp.stdout.strip(), file=sys.stderr)
    except Exception:
        pass
    try:
        vf = run([str(serve), "--verify-pack", "--model", str(model)], check=False)
        if vf.returncode == 0:
            print("verify-pack: OK", file=sys.stderr)
        else:
            print("verify-pack: nonzero (development run will proceed)", file=sys.stderr)
    except Exception:
        pass

    if args.host != "127.0.0.1":
        raise ValueError("This experimental launcher only serves on 127.0.0.1")
    if not port_free(args.host, args.port):
        raise RuntimeError(f"Port {args.port} is not free on {args.host}")

    # Compose argv (development-only; explicit override for unvalidated pack).
    argv = [
        str(serve),
        "--model", str(model),
        "--host", args.host, "--port", str(args.port),
        "--slots", str(args.slots),
        "--max-model-len", str(args.max_model_len),
        "--spec", "auto",
        "--prefix-cache", "on",
        "--prefix-cache-mib", str(args.prefix_cache_mib),
        "--served-model-name", args.served_model_name,
        "--allow-unvalidated-pack",
    ]
    env = os.environ.copy()
    # Lower floor for more headroom without changing system config.
    env.setdefault("REXL3_MEMORY_FLOOR_GIB", "4")

    # Start server (foreground) and poll health from this process. We avoid
    # backgrounding so that Ctrl-C stops only this process and nothing else.
    print("Starting server:", " ".join(argv), file=sys.stderr)
    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(argv, stdout=logf, stderr=logf, env=env, text=True)

        def _term(signum, frame):
            try:
                proc.terminate()
            except Exception:
                pass
        signal.signal(signal.SIGINT, _term)
        signal.signal(signal.SIGTERM, _term)

        base = f"http://{args.host}:{args.port}"
        deadline = time.time() + float(args.health_timeout_s)
        healthy = False
        while time.time() < deadline:
            time.sleep(1.0)
            try:
                # Use curl if present; otherwise consider logs-only readiness.
                curl = shutil.which("curl")
            except Exception:
                curl = None
            if curl:
                try:
                    r = run([curl, "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"{base}/health/ready"], timeout=5)
                    if r.stdout.strip() == "200":
                        healthy = True
                        break
                except Exception:
                    pass
        if healthy:
            print(f"READY at {base} (loopback only). Tail logs: {log_path}", file=sys.stderr)
        else:
            print(f"Did not observe READY within {args.health_timeout_s}s; check logs: {log_path}", file=sys.stderr)
        # Wait until exit (Ctrl-C to stop).
        proc.wait()
        return proc.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)


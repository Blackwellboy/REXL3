#!/usr/bin/env python3
"""
Preflight-only launcher for GLM-5.3-Flash SAGE REXL3 2.48 bpw (single GB10).

- Development-only. Does NOT modify sealed trees or approvals.
- Does NOT start the server and does NOT bypass engine gates.
- Prints a JSON plan (argv/env) that would be used to start a loopback-only
  single-node test, without asserting capacity or changing model/profile locks.
- Use to validate paths, basic inputs, and to record native binary identity.

Usage (preflight only):
  python3 tools/dev/serve_glm53_2p48.py \
    --serve-bin $HOME/rexl3/bin/rexl3-serve \
    --model-dir /path/to/local/snapshot \
    [--host 127.0.0.1] [--port 8955] [--slots 1]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import socket
import shutil
import json as _json


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
    ap.add_argument("--served-model-name", default="GLM-5.3-Flash-SAGE-REXL3-2.48bpw")
    ap.add_argument("--check-port", action="store_true", help="Also report if the port appears free on host")
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

    # Optional read-only probes; ignore failures (not a production seal; does not imply fitness).
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
            print("verify-pack: nonzero (no start attempted by this tool)", file=sys.stderr)
    except Exception:
        pass

    if args.host != "127.0.0.1":
        raise ValueError("This preflight tool plans loopback-only (127.0.0.1) execution")
    port_status = None
    if args.check_port:
        port_status = "free" if port_free(args.host, args.port) else "occupied"

    # Compose a minimal argv plan without altering model/profile limits.
    argv = [
        str(serve),
        "--model", str(model),
        "--host", args.host, "--port", str(args.port),
        "--slots", str(args.slots)
    ]
    plan = {
        "note": "Preflight only; does not start the server; does not bypass engine gates.",
        "serve_bin": str(serve),
        "model_dir": str(model),
        "host": args.host,
        "port": args.port,
        "slots": args.slots,
        "argv": argv,
        "port_status": port_status,
        "native_version": ver.stdout.strip(),
    }
    print(_json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)


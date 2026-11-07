# Experimental dev serve: GLM-5.3-Flash SAGE REXL3 2.48 bpw (single GB10 / Headchef only)

Status: experimental, development-only. This does NOT modify sealed trees, approvals, or production policy, and does not claim qualification. It runs the native REXL3 server directly against the exact 2.48 bpw EXL3 pack.

Scope and constraints

- Host: one NVIDIA GB10 (DGX Spark class), Ubuntu 24.04 ARM64 (Headchef only).
- Keep existing workloads: do not stop other machines; single-node only; binds 127.0.0.1 on a separate unused port.
- No releases, no sealed edits, no doctor/lock bypass in production. This dev flow uses the engine’s documented unvalidated-pack override to boot a non-validated EXL3 pack for local testing only.
- No driver/firmware/kernel changes, no fan/EC/cap changes, no blanket sudo. Python/user-space tools only.
- Exact model: `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` (15 shards), text+vision on. You must have HF access to the gated repo.

What this proves

- Native engine can parse/serve this EXL3 pack on a single GB10 at a conservative context window and one slot, with loopback-only binding, and without broad process kills.
- The run records the actual driver/library versions the binary reports; there is no attempt to fake production pins.

What this does NOT prove

- No production qualification, gates, or floors. No sealed profile or approval is added or altered. This is not a substitute for the promotion process.

Prereqs on Headchef (read-only checks first)

1) Ensure you have the v0.1.0 native engine binary already staged (as reported):
   - `rexl3-serve-0.1.0-799fa750-linux-aarch64-sm121` (sha256 `3e188724f538d25f7d354e34865a2205c266274171f1d000f764fa5093ad4041`)
   - Place it at e.g. `$HOME/rexl3/bin/rexl3-serve` and `chmod 755`.
2) Confirm runtime libraries present:
   - CUDA runtime `cuda-cudart-13-0` 13.0.96-1
   - cuBLAS `libcublas-13-0` 13.1.1.3-1
   - NCCL is not required for single-node.
3) Confirm driver (no change required): `nvidia-smi` shows 580.178.04 (newer than production v0.1.0 pin; that pin is a production gate, not a dev-run gate).

Fetch the exact model pack to a local directory

- You must be granted access to the gated repo.
- Use git-lfs or `hf` CLI; example with git-lfs:

```bash
PACK_DIR=$HOME/models/glm53-2p48
mkdir -p "$PACK_DIR"
cd "$PACK_DIR"
git init && git remote add origin https://huggingface.co/vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw
git fetch --depth=1 origin +refs/heads/main:refs/remotes/origin/main
git checkout origin/main
git lfs pull --include="model-*.safetensors,model.safetensors.index.json,config.json,generation_config.json,tokenizer*.json,processor_config.json,chat_template.jinja"
```

Verify the expected file set and sizes (sanity, not production sealing):

```bash
# Expect 15 shard files and index; rough total ~106,796,830,300 bytes (~99.46 GiB)
python3 - <<'PY'
import os, glob, sys
p=os.environ.get("PACK_DIR") or "."
files=sorted(glob.glob(os.path.join(p, "model-*.safetensors")))
print("shards:", len(files))
total=sum(os.path.getsize(f) for f in files)
if os.path.exists(os.path.join(p,"model.safetensors.index.json")):
  total+=os.path.getsize(os.path.join(p,"model.safetensors.index.json"))
print("bytes_total:", total)
PY
```

Read-only pack validation with the engine (no start yet)

```bash
REXL3_SERVE=$HOME/rexl3/bin/rexl3-serve
$REXL3_SERVE --version
# Expect to print version, commit, cuda_driver, cudart, cublas, nvcc, sm_121.

# Print pack fingerprints (informational)
$REXL3_SERVE --print-fingerprints --model "$PACK_DIR" || true

# Verify pack container integrity (CRC/structure); returns nonzero on structural issues
$REXL3_SERVE --verify-pack --model "$PACK_DIR" || true
```

Start a conservative single-node dev server (loopback-only)

- One slot, low initial window (131,072) to keep residency margin; prefix cache small (1024 MiB).
- Bind 127.0.0.1 on a free port (8955 recommended). Do not expose publicly.
- Use the engine’s documented dev override: `--allow-unvalidated-pack`.

```bash
PORT=8955
HOST=127.0.0.1
REXL3_MEMORY_FLOOR_GIB=4 \
$REXL3_SERVE \
  --model "$PACK_DIR" \
  --host "$HOST" --port "$PORT" \
  --slots 1 \
  --max-model-len 131072 \
  --spec auto \
  --prefix-cache on --prefix-cache-mib 1024 \
  --served-model-name GLM-5.3-Flash-SAGE-REXL3-2.48bpw \
  --allow-unvalidated-pack
```

Safety notes

- The server binds 127.0.0.1 only; no TLS, no auth. Do not route externally. Use SSH tunnels if remote access is needed.
- This process owns only its PID; no broad/system kills are performed. Stop with Ctrl‑C in the same terminal.

Smoke requests (second terminal)

```bash
BASE=http://127.0.0.1:8955/v1
curl -s $BASE/models | python3 -m json.tool
curl -s $BASE/chat/completions -H 'content-type: application/json' -d '{
  "model": "GLM-5.3-Flash-SAGE-REXL3-2.48bpw",
  "messages": [{"role":"user","content":"Reply with the single word: ready"}],
  "max_tokens": 256,
  "chat_template_kwargs": {"reasoning_effort":"low"}
}' | python3 -m json.tool
```

If memory refusal occurs

- Lower the requested window (e.g., 65536), keep `--slots 1`, and ensure no large concurrent GPU tenants. Do not change the device driver or system configuration.

Shutdown and rollback

- Ctrl‑C to stop the server; the process exits cleanly.
- Remove the temporary staging directory if desired; no system-level changes were made.

Appendix: Narrow idle check before starting (optional)

```bash
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU busy; start later to avoid interference."
  exit 1
fi
```

Known differences vs production

- Production doctor enforces strict native version pins (driver/cuBLAS). This dev flow records actual versions and uses loopback-only scope, without modifying sealed policies or trees. No qualification claims are made.


# Experimental dev preflight: GLM-5.3-Flash SAGE REXL3 2.48 bpw (single GB10 / Headchef only)

Status: experimental, development-only. This does NOT modify sealed trees, approvals, or production policy, and does not claim qualification. It does not start the server or bypass engine gates. It performs preflight checks and prints a prospective command plan only.

Scope and constraints

- Host: one NVIDIA GB10 (DGX Spark class), Ubuntu 24.04 ARM64 (Headchef only).
- Keep existing workloads: do not stop other machines; single-node only; intended loopback bind when/if executed outside this doc.
- No releases, no sealed edits, no doctor/lock bypass in production. No use of any flag that disables/relaxes capability, fingerprint, or memory refusal.
- No driver/firmware/kernel changes, no fan/EC/cap changes, no blanket sudo. Python/user-space tools only.
- Exact model: `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` (15 shards), text+vision on.

Not proven here

- No production qualification, gates, or floors. No sealed profile or approval is added or altered. This is not a substitute for the promotion process. No readiness or serving claims are made. No GPU tests are run by this document.

Prereqs on Headchef (read-only checks only)

1) Ensure you have the v0.1.0 native engine binary already staged (as reported in the v0.1.0 release):
   - `rexl3-serve-0.1.0-799fa750-linux-aarch64-sm121` (sha256 `3e188724f538d25f7d354e34865a2205c266274171f1d000f764fa5093ad4041`)
   - Place it at e.g. `$HOME/rexl3/bin/rexl3-serve` and `chmod 755`.
2) Confirm runtime libraries present:
   - CUDA runtime `cuda-cudart-13-0` 13.0.96-1
   - cuBLAS `libcublas-13-0` 13.1.1.3-1
   - NCCL is not required for single-node; two-node requires it (see citation below).
3) Confirm driver (no change required): `nvidia-smi` shows 580.178.04 (newer than production v0.1.0 pin; that pin is a production gate, not a dev-run gate).

Fetch the exact model pack to a local directory (when network is available)

- Use git and git-lfs; pin an immutable commit SHA (do not assume `main`). Example:

```bash
PACK_DIR=$HOME/models/glm53-2p48
mkdir -p "$PACK_DIR"
cd "$PACK_DIR"
git init
git remote add origin https://huggingface.co/vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw
# Replace <COMMIT> with the exact immutable commit SHA you record.
git fetch --depth=1 origin +refs/heads/main:refs/remotes/origin/main
git checkout <COMMIT>
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

Read-only pack validation with the engine (no start; no gate bypass)

```bash
REXL3_SERVE=$HOME/rexl3/bin/rexl3-serve
$REXL3_SERVE --version
# Expect to print version, commit, cuda_driver, cudart, cublas, nvcc, sm_121.

# Print pack fingerprints (informational)
$REXL3_SERVE --print-fingerprints --model "$PACK_DIR" || true

# Verify pack container integrity (CRC/structure); returns nonzero on structural issues
$REXL3_SERVE --verify-pack --model "$PACK_DIR" || true
```

Preflight-only: print a prospective loopback plan (no start)

- Bind 127.0.0.1 on a free port (e.g., 8955). Do not expose publicly.
- This step only prints a JSON plan; it does NOT start the server and does NOT bypass gates.

```bash
python3 tools/dev/serve_glm53_2p48.py \
  --serve-bin "$REXL3_SERVE" \
  --model-dir "$PACK_DIR" \
  --host 127.0.0.1 --port 8955 --slots 1 --check-port
```

No smoke requests (no server running)

This workflow does not make serving or readiness claims. No GPU tests are run here.

Shutdown and rollback

- No processes started by this tool. Remove the temporary staging directory if desired; no system-level changes are made.

Appendix: Narrow idle check before a future start (optional; not executed here)

```bash
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU busy; start later to avoid interference."
  exit 1
fi
```

Documented flags and gates (citations)

- OpenAI-compatible server CLI (supported flags): project README “Run your own model or recipe (advanced)”.

```192:205:README.md
rexl3-serve --model <model-dir> [--draft <draft-dir>] --host 127.0.0.1 --port 8889 \
  --slots 8 --max-model-len 262144 [--kv-dtype fp16|q4|int8] [--spec auto|mtp|copy|none] \
  [--prefix-cache on|off] [--prefix-cache-mib N] --served-model-name NAME
```

- Pack fingerprints and unvalidated-pack gate (documented; NOT used by this document):

```200:212:README.md
Only packs on the validated list boot without extra flags; --verify-pack checks the shard
CRC-32s, --print-fingerprints prints the fingerprints, and --print-crc32 prints shard CRCs.
Unvalidated pack override: a pack that is not on the validated list is refused unless you
explicitly pass --allow-unvalidated-pack. This is a deliberate, explicit override — never a default.
```

- Doctor strict native-version policy (driver/cuBLAS pins):

```6:16:runtimes/glm-5.3/glm-5.3-flash/v1/scripts/doctor.py
Native --version is intentionally strict for *all* source-defined fields,
including the runtime CUDA driver/cuBLAS versions: newer is not automatically
qualified. Never edit the lock or change drivers to suppress that failure.
```

- NCCL dependency for two-node; single-node does not use it:

```168:178:docs/packs/glm-5.3-flash.md
dependencies.lock.json is the DeepSeek-V4.1 two-node lock ... because a pair without libnccl.so.2 exits at boot.
A single node installs NCCL too and does not use it.
```


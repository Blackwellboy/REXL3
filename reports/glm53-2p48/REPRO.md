GLM-5.3-Flash 2.48bpw header-only audit (public-source, CPU-only)

Scope:
- Public Hugging Face repo `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw`, pinned revision `828837bf87c4a0b4855491dcfb2bc2977a455e5d`.
- Public REXL3 v0.1.0 sources (afc9/799fa) cited in PR body; no sealed edits or gate relaxations; no serving.
- CPU-only; strict HTTP Range reads for safetensors headers only (first 8 bytes + header region); no shard bodies.
- No CRC32 and no authoritative fingerprint; no GPU tests; no deployment.

Reproduce on a host with Python 3.11+ and outbound HTTPS:

```bash
cd /workspace
python3 tools/glm248_header_audit.py \
  --repo vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw \
  --rev 828837bf87c4a0b4855491dcfb2bc2977a455e5d \
  --out artifacts/glm53-2p48
```

Outputs (immutable, copyable):
- `artifacts/glm53-2p48/<rev>/hf_revision.json`: HF API metadata (blobs=true) with LFS ids and sizes.
- `artifacts/glm53-2p48/<rev>/manifest.tsv`: file, LFS sha256, size for `config.json`, all 15 shards, and `model.safetensors.index.json`.
- `artifacts/glm53-2p48/<rev>/config.json` and `.sha256.txt`
- `artifacts/glm53-2p48/<rev>/model.safetensors.index.json` and `.sha256.txt`
- For each shard `model-000NN-of-00015.safetensors`:
  - `.first8.bin` and `.first8.http` (status 206 + Content-Range)
  - `.header.json` (header region only) and `.header.http`
- `artifacts/glm53-2p48/<rev>/range_meta.json`: per-shard file_size, header_len, calc_data_bytes, and check == (file_size - 8 - header_len)
- `artifacts/glm53-2p48/<rev>/summary.json` and `summary.txt`: high-level counts and checks

What is validated vs not:
- Validated (header-only):
  - Header provenance (HTTP 206 with Content-Range for both ranges).
  - Total per-shard data bytes from headers equals file_size - 8 - header_len (from Content-Range).
  - Index mapping: every `weight_map` entry appears in headers; duplicates map to a single chosen shard.
  - EXL3 group acceptance (header-only, per v0.1.0 public source): trellis->bits, K/N multiples of 128, scale dtypes/shapes, marker presence consistency.
  - GLM text loader restrictions (header-only): integer bits, fp16 scales (suh/svh), no bias.
  - Expert-bank uniformity (by layer and kind): uniform K/N/bits/codebook/scales across experts.
  - MTP/NextN presence keys in index; `eh_proj` dims match config hidden size (2*h, h) where present.
- Not validated here:
  - CRC32 of shards (requires full bodies).
  - Authoritative engine fingerprint (`rexl3-serve --print-fingerprints`).
  - Any GPU-bound runtime behaviour, binding, decoding, or performance.

Notes:
- This audit is evidence for header/layout compatibility only. It is not a qualification or a serving claim and does not change any validated lists or gates.


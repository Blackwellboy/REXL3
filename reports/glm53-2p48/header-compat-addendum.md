GLM-5.3-Flash 2.48bpw header/layout compatibility addendum (public v0.1.0)

Scope and constraints
- Public source only: REXL3 v0.1.0 engine sources (commits 799fa750 and afc9b5df) inspected locally. No sealed tree edits or gate relaxations.
- Pack: `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` pinned revision `828837bf87c4a0b4855491dcfb2bc2977a455e5d`.
- CPU-only: strict HTTP range reads for safetensors headers; no weight bodies downloaded; no serving or GPU tests.
- This is not a validation or serving claim. GLM pack-gate remains enforced.

Pinned artifacts (reproducible)
- See `reports/glm53-2p48/REPRO.md` for one-command repro and artifact layout.
- Artifact base: `artifacts/glm53-2p48/828837bf87c4a0b4855491dcfb2bc2977a455e5d/`
  - `hf_revision.json`: full HF metadata (blobs=true) — immutable LFS object ids + sizes
  - `manifest.tsv`: file, LFS sha256, size for config, index, and all 15 shards
  - `config.json` (sha256 in `.sha256.txt`)
  - `model.safetensors.index.json` (sha256 in `.sha256.txt`)
  - Per-shard: `.first8.bin` + `.first8.http` (206 + Content-Range), `.header.json` + `.header.http`, with `range_meta.json`
  - `summary.txt` (human) and `summary.json` (machine)

Header/layout checks (public-source rules, header-only)
- EXL3 group acceptance rules (799fa public source, header-only) matched for all text groups:
  - Trellis dims -> bits_twice ∈ {2,3,4,5,6,7,8,10,12,14,16}; K,N multiples of 128; scales present and correctly typed/shaped.
  - Codebook presence (`mul1`) consistent; marker value not verified from bytes (header-only).
- GLM loader restrictions for text side (799fa):
  - Integer bits only (no fractional widths), fp16 scales (suh/svh), no bias — satisfied by all accepted text groups.
- Expert-bank uniformity:
  - For every `layers.<L>.mlp.experts.*.(down_proj|gate_proj|up_proj)`, K/N/bits/codebook/scales were uniform across experts per layer.
- Index consistency:
  - Every `weight_map` entry was present in exactly one shard header; duplicates resolved to the index’s chosen shard; no unindexed header names.
- NextN/MTP presence:
  - MTP-related keys are present (e.g., `layers.45.eh_proj.*`, `layers.45.shared_head.norm.weight`).
  - With `hidden_size=4096` from `config.json`, `eh_proj` dims match (K=8192, N=4096).

Memory/provenance sanity
- For each shard: `sum(dtype_size(shape))` from header equals `file_size - 8 - header_len` (from HTTP Content-Range and first8), see `range_meta.json`.
- Aggregate data bytes across all shards: ~99.41 GiB, matching the HF LFS sizes minus headers.
- All HTTP header fetches were strict 206 Range responses; headers saved for provenance.

Public-source code citations (v0.1.0)
- CLI flags exist in server (both 799fa/afc9): `--print-fingerprints`, `--print-crc32`, `--verify-pack`; non-serving flags.
- GLM pack-gate enforced; DeepSeek/Qwen-MoE special allowlist does not apply to GLM.
- EXL3 acceptance and GLM text restrictions implemented as cited below.

Limitations (what is not proven here)
- No CRC32 (requires full shard bodies).
- No authoritative engine fingerprint (requires engine’s exact canonicalization over accepted groups).
- No GPU binding/decoding/perf; no serving claims; no gate changes.

Summary (from `summary.txt`)

```
HF repo: vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw
Revision: 828837bf87c4a0b4855491dcfb2bc2977a455e5d
Manifest rows: 17
Index weight_map entries: 150894
Range/memory sums: ok; total data 99.41 GiB across 15 shards
Groups accepted: 37420; refused: 0; GLM text violations: 0
Non-uniform banks: 0; MTP keys: 7; hidden=4096 vocab=154880 eh_proj_ok=True
Index missing: 0; index mismatches: 0; unindexed headers: 0
```

Refusal classification correction (audit logic aligned to public engine)
- Previous audit iteration counted 135 “refused” due to prefixes that had EXL3-related tensors but no `.trellis`. The public engine’s scan ignores such prefixes (not counted as groups) unless the trellis itself is malformed. Audit logic now mirrors that: skip non‑trellis prefixes (not refusals). Current true group refusals: 0.
- Non-group prefixes with parts (no trellis): 135 (e.g., `model.language_model.layers.11.self_attn.indexer.k_norm`, `...layers.15...k_norm`, `...layers.19...k_norm`, `...layers.23...k_norm`, `...layers.27...k_norm`). These are ordinary tensors or auxiliary parts, not EXL3 linears; loader maps/filters them via inventory, not through EXL3 group acceptance.

Citations (source lines, v0.1.0 public)
- EXL3 group acceptance (header-only parser):
  - `crates/exl3/src/checkpoint.rs`
- GLM text loader restrictions (integer bits, fp16 scales, no bias):
  - `crates/model/src/families/glm/bind.rs` (`unsupported()`)
- Pack validated-list gate and DeepSeek allowance:
  - `crates/model/src/serve.rs` (`check_pack_dirs`, `check_packs`)
- CLI non-serving flags:
  - `crates/server/src/main.rs` (`--print-fingerprints`, `--print-crc32`, `--verify-pack`)


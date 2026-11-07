# Source compatibility findings — GLM-5.3-Flash SAGE REXL3 2.48 bpw (public v0.1.0 engine)

Purpose

- Determine from public v0.1.0 corresponding source whether the native engine can be legitimately adapted to support the exact pack `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` (no sealed edits, no bypasses), and identify missing pieces if any.

Public source archives (downloaded on this VM)

- Tag: `v0.1.0`
- General engine source: `rexl3-engine-0.1.0-afc9b5df-source.tar.gz`  
  Immutable URL `https://github.com/REXL3/REXL3/releases/download/v0.1.0/rexl3-engine-0.1.0-afc9b5df-source.tar.gz`  
  SHA-256 (computed here): `ade2a05b0629cbbbc8921ee945cad18ffff1e899a3f9c83dd74b24504fe7a046`  
  `SOURCE_BUILD.md` commit: `afc9b5df...`
- GLM pack engine source: `glm53-flash-2.25bpw-sm121-v1-engine-source.tar.gz`  
  Immutable URL `https://github.com/REXL3/REXL3/releases/download/v0.1.0/glm53-flash-2.25bpw-sm121-v1-engine-source.tar.gz`  
  `SOURCE_BUILD.md` commit: `799fa750...`

Distinction: the GLM sealed pack was frozen on `799fa750...`; adaptation should be reasoned against that source.

Evidence in source: EXL3 pack loader (mixed-K, per-group K/N, codebooks)

- `crates/exl3/src/checkpoint.rs` parses pack headers (no weight reads) and admits/refuses EXL3 groups and plain tensors. Key points:
  - Per-group K and N via trellis shape; bits from trellis last dimension; codebook `inst3`/`mcg`/`mul1`; supports `.suh/.su` and `.svh/.sv`.
  - Index-aware scan (`scan_indexed`) honors `model.safetensors.index.json::weight_map` and records duplicates/strays.
  - No assumption on shard count; shards resolved via index or scan.

```text
.../crates/exl3/src/checkpoint.rs (selected lines)
let bits = Bits::from_trellis_last(tr.shape[2])...;   // variable bit-width per group
let (k, n) = (16 * tr.shape[0], 16 * tr.shape[1]);    // K,N from trellis
... in_scales: "suh" or "su" with shapes [k] or [k/16]
... out_scales: "svh" or "sv" with shapes [n] or [n/16]
... Codebook::Inst3 / ::Mcg / ::Mul1 and marker checks
... index_files(...) and scan_indexed(...) honor weight_map placement
```

Evidence in source: GLM family + NextN (MTP) speculation

- GLM family implementation present, including speculation:
  - `crates/model/src/families/glm/spec.rs`: MTP/copy drafting policies, arbiter model, depths, cost tables, “drafted == serial” contract.
  - GLM pack layout, mapping, and plan in `crates/model/src/families/glm/{inventory.rs,plan.rs,load.rs,weights.rs}`.
  - CUDA kernels for EXL3 linears and GLM P2 prefill under `crates/cuda/csrc/*` (e.g. `exl3_big.cu`, `exl3_linear.cu`, `glm_p2_dense.cu`).

Evidence in source: pack fingerprint and validated-list gate

- Validated pack acceptance is centralized:
  - `crates/model/src/fingerprint.rs`: `VALIDATED_TEXT` (text-only packs, includes GLM 2.25 bpw fingerprint), plus `VALIDATED_SHARDS` (pinned shard names and LFS ids + CRC-32).
  - `crates/model/src/serve.rs`: `check_packs` computes fingerprints and refuses when neither target nor (optional) draft are on the validated list; with an operator override the server marks it unvalidated. Production doctor remains strict on native versions (driver/cuBLAS).
  - `crates/server/src/main.rs`: CLI surfaces `--print-fingerprints` and `--verify-pack`, and declares `--print-crc32` among non‑serving flags (prints and exits).

```text
.../crates/model/src/fingerprint.rs
pub const VALIDATED_TEXT: &[ValidatedText] = &[
  ("GLM-5.3-Flash-EXL3-2.25bpw-sm121 (text)", "sha256:96a51f..."),
  ...
];
pub const VALIDATED_SHARDS: &[(&str, &[PinnedShard])] = &[ ... ];
```

NCCL note (single node vs TP=2)

- Two-node TP=2 needs NCCL (`crates/cuda/src/nccl.rs` referenced by GLM TP boot paths). The GLM single-node profile does not use NCCL. The pack sheet notes align with this.

Feasibility vs. validated support (correction)

- The public source at `799fa750...` contains code paths to parse and plan GLM5Next EXL3 mixed‑K packs and expose fingerprint/CRC reporting. This indicates plausible code‑level compatibility.
- It does NOT constitute qualification or production acceptance of the exact 2.48 bpw pack. Adding entries to `VALIDATED_TEXT/SHARDS` represents a maintainer’s promotion decision after passing qualification; it is not a substitute for it and is not proposed here.

What is missing before any consideration of validation entries

- The pack fingerprint and pinned shard manifests for the exact `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` revision (15 shards).
  - Computing these requires local access to the pack directory to run:
    - `rexl3-serve --print-fingerprints --model <DIR>` (fingerprint + validated/no line)
    - `rexl3-serve --print-crc32 --model <DIR>` (per-shard CRC-32; for `VALIDATED_SHARDS`)
  - On Headchef, large transfers are currently blocked; no weights are present here. Without the pack, we cannot derive the immutable fingerprint or the CRC-32 table.

Non-bypass experimental development recognition path (no production gates altered)

1) Resolve and pin the pack’s immutable HF commit SHA (operator step; no download here).
2) On a host with the pack present (e.g., Headchef when egress is allowed or via offline copy):
   - Run `rexl3-serve --print-fingerprints --model DIR` to obtain the `sha256:...` fingerprint.
   - Run `rexl3-serve --print-crc32 --model DIR` to collect `(shard name, CRC-32)`; record LFS object SHA-256 ids from HF file listings for the same names.
3) Build a separate, reviewed development validation path (not production acceptance):
   - CPU‑only tests (compile-time/unit): CLI non‑serving flag parsing; synthetic EXL3 header scanners (existing tests cover mixed‑K/codebooks).
   - With local headers: header‑only scans of the exact pack (no weight data reads) to ensure no refused groups and a consistent index mapping.
4) Only after completing the full qualification suite (exactness/speculation/prefix‑cache/vision/performance) on `799fa750...` would a maintainer-owned promotion add validated pins. That change is out of scope here.

Where confidence stops

- No GPU binary build or run in this VM; no model weights; no fingerprint values collected here.
- We did not alter or propose altering production policy/doctor, and did not use or recommend any bypass flags in this PR.

Notes on HF metadata availability (without weights)

- Config and file listings are accessible via public APIs; immutable commit resolution should be recorded by the operator. The index JSON is large (we saw ~15.7 MB) and our connector caps `cat` to 80KB. Fingerprints depend on safetensors headers (group trellises/codebooks), so headers must be locally accessible; config/index alone are insufficient.

Appendix: additional code pointers (public source)

- GLM loader/planner: `crates/model/src/families/glm/{inventory.rs,plan.rs,load.rs,weights.rs}`
- NextN (MTP) policy: `crates/model/src/families/glm/spec.rs`
- EXL3 kernels: `crates/cuda/csrc/exl3_*.cu`, `exl3_grouped.cuh`, `exl3_linear_decode.cuh`
- Fingerprint gate: `crates/model/src/fingerprint.rs`, `crates/model/src/serve.rs`
- CLI flags surfaced: `crates/server/src/main.rs` (includes `--print-fingerprints`, `--verify-pack`, and `--print-crc32` in non‑serving flags)

DeepSeek validated‑pack gate behavior (public source; for completeness)

- Both `afc9b5df` and `799fa750` sources include an additional condition in `check_pack_dirs` that sets the `allow_unvalidated` parameter when the target `config.json` text contains `deepseek_v41` (or `qwen3_5_moe`). This causes `check_packs` to accept such packs as unvalidated without requiring the operator flag. GLM (`glm5_next`) does not match this condition and remains subject to the validated‑pack check (or an explicit operator flag).

```454:473:/tmp/tmp.qZ1hKOZ6hC/crates/model/src/serve.rs
let is_dsv41 = std::fs::read_to_string(o.model_dir.join("config.json"))
    .is_ok_and(|s| s.contains("deepseek_v41") || s.contains("qwen3_5_moe"));
check_packs(&target, draft.as_ref(), o.allow_unvalidated_pack || is_dsv41)...
```

- This observation does not change our GLM plan: GLM 2.48 bpw remains gated by validated fingerprint lists or an explicit operator override. No bypass is suggested or used by this PR.

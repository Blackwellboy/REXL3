# Source compatibility findings — GLM-5.3-Flash SAGE REXL3 2.48 bpw (public v0.1.0 engine)

Purpose

- Determine from public v0.1.0 corresponding source whether the native engine can be legitimately adapted to support the exact pack `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` (no sealed edits, no bypasses), and identify missing pieces if any.

Public source archive (downloaded on this VM)

- Tag: `v0.1.0`
- Asset: `rexl3-engine-0.1.0-afc9b5df-source.tar.gz`
- Immutable URL: `https://github.com/REXL3/REXL3/releases/download/v0.1.0/rexl3-engine-0.1.0-afc9b5df-source.tar.gz`
- SHA-256 of the gzip tarball (computed here): `ade2a05b0629cbbbc8921ee945cad18ffff1e899a3f9c83dd74b24504fe7a046`
- Extracted `SOURCE_BUILD.md` states engine source commit `afc9b5df...` and toolchain 1.89.0.

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
  - `crates/server/src/main.rs`: CLI surfaces `--print-fingerprints` and `--verify-pack`. (Not used in this PR’s preflight; present for manual fingerprint/shard pin derivation.)

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

Public-source feasibility conclusion

- The public v0.1.0 engine source contains:
  - EXL3 trellis/group scanner supporting mixed-K and codebooks, index-aware, shard-agnostic.
  - GLM5-Next backend, including NextN/MTP speculation, P2 prefill, and GLM-specific kernels/plans.
  - A validated-pack gate implemented in public code (`fingerprint.rs` + `serve.rs`) that can be extended with new fingerprints and pinned shards.
- Therefore, adapting the engine to support the exact 2.48 bpw pack legitimately is feasible within the public source by:
  1) computing the pack’s fingerprint with this engine (`rexl3-serve --print-fingerprints --model DIR`), and
  2) adding it to `VALIDATED_TEXT` with its pinned shards under `VALIDATED_SHARDS` (file names, LFS SHA-256 ids, and CRC-32s), then
  3) rebuilding from `SOURCE_BUILD.md`. No loader/kernel architecture change is indicated by the code.

What is missing to complete the adaptation now

- The pack fingerprint and pinned shard manifests for the exact `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` revision (15 shards).
  - Computing these requires local access to the pack directory to run:
    - `rexl3-serve --print-fingerprints --model <DIR>` (fingerprint + validated/no line)
    - `rexl3-serve --print-crc32 --model <DIR>` (per-shard CRC-32; for `VALIDATED_SHARDS`)
  - On Headchef, large transfers are currently blocked; no weights are present here. Without the pack, we cannot derive the immutable fingerprint or the CRC-32 table.

Non-bypass minimal implementation plan (public-source based)

1) Resolve and pin the pack’s immutable HF commit SHA (operator step; no download here).
2) On a host with the pack present (e.g., Headchef when egress is allowed or via offline copy):
   - Run `rexl3-serve --print-fingerprints --model DIR` to obtain the `sha256:...` fingerprint.
   - Run `rexl3-serve --print-crc32 --model DIR` to collect `(shard name, CRC-32)`; record LFS object SHA-256 ids from HF file listings for the same names.
3) In this public source:
   - Edit `crates/model/src/fingerprint.rs`:
     - Append `(\"vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw (text)\", \"sha256:<FINGERPRINT>\")` to `VALIDATED_TEXT`.
     - Append a new block to `VALIDATED_SHARDS` keyed by `\"sha256:<FINGERPRINT>\"`, listing `(\"model-000xx-of-00015.safetensors\", \"<LFS SHA256>\", <CRC32>)` for all shards present in `model.safetensors.index.json`.
   - Optional: add a compile-time-only unit test that asserts `validated_text("<FINGERPRINT>").is_some()` (does not require the real pack).
4) Rebuild per `SOURCE_BUILD.md` (no GPU build required in CI here; the native GPU build step is for the operator environment).
5) Operator-only, later: run doctor and serving lifecycle under a development profile (not part of this PR).

Where confidence stops

- No GPU binary build or run in this VM; no model weights; no fingerprint values collected here.
- We did not alter or propose altering production policy/doctor, and did not use or recommend any bypass flags in this PR.

Notes on HF metadata availability

- File lists for the target repo are accessible via the HF APIs/tools; immutable commit resolution should be recorded by the operator. This PR does not embed a commit id and does not assume `main`.

Appendix: additional code pointers (public source)

- GLM loader/planner: `crates/model/src/families/glm/{inventory.rs,plan.rs,load.rs,weights.rs}`
- NextN (MTP) policy: `crates/model/src/families/glm/spec.rs`
- EXL3 kernels: `crates/cuda/csrc/exl3_*.cu`, `exl3_grouped.cuh`, `exl3_linear_decode.cuh`
- Fingerprint gate: `crates/model/src/fingerprint.rs`, `crates/model/src/serve.rs`
- CLI flags surfaced: `crates/server/src/main.rs`

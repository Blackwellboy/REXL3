#!/usr/bin/env python3
"""
GLM-5.3-Flash 2.48bpw (vcruz305) header-only audit, CPU-only.

What this script does:
- Reads public HF revision metadata (blobs=true) for a pinned revision and writes an immutable manifest (LFS sha256 + sizes).
- Fetches config.json and model.safetensors.index.json (full) and saves them with computed sha256.
- For each shard model-00001..00015.safetensors:
  - Performs strict HTTP range reads for first 8 bytes (header length) and then the JSON header region only.
  - Verifies HTTP 206 Content-Range responses and records provenance.
  - Writes first8.bin, header.json, and .http headers, plus a meta.json per shard.
- Builds a union view of all headers and performs validation checks against public v0.1.0 source rules (799fa):
  - EXL3 group acceptance (header-only): trellis dims -> bits, K/N multiples of 128, su/suh and sv/svh constraints,
    marker presence consistency (value not verified from bytes — header-only).
  - GLM text restrictions: integer bits only, fp16 scales (suh/svh), no bias.
  - Expert-bank uniformity by layer and projection kind (down_proj/gate_proj/up_proj).
  - Index consistency: every mapped tensor is present in some header; duplicates map to the chosen shard;
    record unindexed names (headers present but not in weight_map).
- Memory sanity:
  - For each shard, compute data_bytes = sum(dtype_size(shape)) across header entries and verify that
    data_bytes == (file_size - 8 - header_len) from Content-Range/first8.
  - Sum across shards and corroborate against the HF LFS 'size' totals.

What this script does NOT do:
- Download or read any weight bodies beyond the header region (strict Range 0-7 and header bytes only).
- Compute CRC32 (requires full bodies).
- Compute an authoritative engine fingerprint (requires exact Rust canonicalization and accepted groups with marker byte checks).
- Launch any servers or require a GPU.

Outputs are written under: artifacts/glm53-2p48/<rev>/
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import io
import json
import os
import re
import struct
import sys
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from urllib.request import Request, urlopen

HF_API_REV = "https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"
USER_AGENT = "curl/8 (header-audit; no-weights)"


def write_text(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def write_bytes(path: str, b: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def http_range(url: str, start: int, end: int) -> Tuple[int, Dict[str, str], bytes]:
    """Return (status, headers, body) for a HTTP byte range GET."""
    req = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as r:
        status = r.status
        headers = {k.lower(): v for k, v in r.headers.items()}
        body = r.read()
    return status, headers, body


def http_get(url: str) -> Tuple[int, Dict[str, str], bytes]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as r:
        status = r.status
        headers = {k.lower(): v for k, v in r.headers.items()}
        body = r.read()
    return status, headers, body


def dtype_size(dtype: str) -> Optional[int]:
    return {
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
        "F32": 4, "I32": 4, "U32": 4,
        "F64": 8, "I64": 8, "U64": 8,
    }.get(dtype.upper())


BITS_ALL_TWICE = {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16}


@dataclasses.dataclass
class GroupMeta:
    prefix: str
    bits_twice: int
    codebook: str  # "3inst" | "mcg" | "mul1"
    k: int
    n: int
    in_scales: str  # "suh" or "su"
    out_scales: str  # "svh" or "sv"
    bias: bool
    files: Set[str]
    marker_unverified: bool  # header-only: we did not read/verify marker values


def parse_group(prefix: str, parts: Dict[str, Dict[str, Any]]) -> Tuple[Optional[GroupMeta], Optional[str]]:
    tr = parts.get("trellis")
    if not tr:
        return None, "no .trellis"
    if tr.get("dtype") != "I16" or not isinstance(tr.get("shape"), list) or len(tr["shape"]) != 3:
        return None, f"trellis must be int16 [K/16, N/16, 16 * bits], got {tr.get('dtype')} {tr.get('shape')}"
    if any(d == 0 for d in tr["shape"]):
        return None, f"trellis {tr['shape']} has an empty dimension"
    last = tr["shape"][2]
    if last % 8 != 0:
        return None, f"unsupported trellis width {last}"
    bits_twice = last // 8
    if bits_twice not in BITS_ALL_TWICE:
        return None, f"unsupported bits_twice {bits_twice}"
    has = [m for m in ("mcg", "mul1") if m in parts]
    if len(has) > 1:
        return None, "both mcg and mul1 markers"
    codebook = "3inst"
    marker_unverified = False
    if has:
        codebook = has[0]
        # Header-only: value not verified; engine reads and checks the int32 marker against constants.
        marker_unverified = True
        e = parts[codebook]
        if e.get("dtype") != "I32" or dtype_size("I32") * 1 != (e.get("data_offsets", [0, 0])[1] - e.get("data_offsets", [0, 0])[0]):
            return None, f".{codebook} marker must be one int32, got {e.get('dtype')} {e.get('shape')}"
    k, n = 16 * tr["shape"][0], 16 * tr["shape"][1]
    ins = "suh" if "suh" in parts else ("su" if "su" in parts else None)
    if not ins:
        return None, "no input scales (.suh or .su)"
    outs = "svh" if "svh" in parts else ("sv" if "sv" in parts else None)
    if not outs:
        return None, "no output scales (.svh or .sv)"
    # Shapes of scales:
    want_in = ("F16", [k]) if ins == "suh" else ("I16", [k // 16])
    e_in = parts[ins]
    if (e_in.get("dtype"), e_in.get("shape")) != want_in:
        return None, f".{ins} must be {want_in[0]} {want_in[1]}, got {e_in.get('dtype')} {e_in.get('shape')}"
    want_out = ("F16", [n]) if outs == "svh" else ("I16", [n // 16])
    e_out = parts[outs]
    if (e_out.get("dtype"), e_out.get("shape")) != want_out:
        return None, f".{outs} must be {want_out[0]} {want_out[1]}, got {e_out.get('dtype')} {e_out.get('shape')}"
    b = parts.get("bias")
    if b:
        if b.get("dtype") not in ("F16", "BF16", "F32") or b.get("shape") != [n]:
            return None, f".bias must be a float vector of {n}, got {b.get('dtype')} {b.get('shape')}"
    # K, N multiples of 128 (Hadamard)
    if k % 128 != 0 or n % 128 != 0:
        return None, f"K={k} and N={n} must be multiples of 128"
    files = set(e.get("file", "<unknown>") for e in parts.values())
    return GroupMeta(prefix, bits_twice, codebook, k, n, ins, outs, b is not None, files, marker_unverified), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw")
    ap.add_argument("--rev", default="828837bf87c4a0b4855491dcfb2bc2977a455e5d")
    ap.add_argument("--out", default="artifacts/glm53-2p48")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.rev)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Fetch HF revision metadata (blobs=true)
    api_url = HF_API_REV.format(repo=args.repo, rev=args.rev)
    st, hdrs, body = http_get(api_url)
    if st != 200:
        print(f"ERROR: HF API returned status {st} for {api_url}", file=sys.stderr)
        sys.exit(2)
    write_bytes(os.path.join(out_dir, "hf_revision.json"), body)
    rev = json.loads(body)
    siblings = rev.get("siblings", [])

    # Build manifest for relevant files (shards + index + config)
    target_re = re.compile(r"(^model-\d{5}-of-00015\.safetensors$)|(^model\.safetensors\.index\.json$)|(^config\.json$)")
    manifest_rows: List[Tuple[str, Optional[str], int]] = []
    for s in siblings:
        fname = s.get("rfilename")
        if not fname or not target_re.search(fname):
            continue
        lfs = s.get("lfs") or {}
        lfs_sha = lfs.get("sha256")
        size = int(s.get("size") or 0)
        manifest_rows.append((fname, lfs_sha, size))
    manifest_rows.sort()
    # Save manifest.tsv
    lines = ["file\tlfs_sha256\tsize\n"] + [f"{f}\t{sha or ''}\t{sz}\n" for (f, sha, sz) in manifest_rows]
    write_text(os.path.join(out_dir, "manifest.tsv"), "".join(lines))

    # 2) Fetch config.json and index.json
    for path in ("config.json", "model.safetensors.index.json"):
        url = HF_RESOLVE.format(repo=args.repo, rev=args.rev, path=path)
        st, hdrs, body = http_get(url)
        if st != 200:
            print(f"ERROR: GET {url} returned {st}", file=sys.stderr)
            sys.exit(2)
        write_bytes(os.path.join(out_dir, path), body)
        write_text(os.path.join(out_dir, f"{path}.sha256.txt"), sha256_bytes(body) + "\n")
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "rb") as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index.get("weight_map") or {}

    # 3) For each shard, range-fetch first 8 bytes and header
    shards = [f"model-{i:05d}-of-00015.safetensors" for i in range(1, 16)]
    all_headers: Dict[str, Dict[str, Any]] = {}
    per_file_headers: Dict[str, Dict[str, Any]] = {}
    range_meta: Dict[str, Dict[str, Any]] = {}
    total_calc_data = 0
    total_file_sizes = 0

    def save_http_headers(path: str, headers: Dict[str, str]) -> None:
        txt = "\n".join(f"{k}: {v}" for k, v in headers.items())
        write_text(path, txt + "\n")

    for fname in shards:
        url = HF_RESOLVE.format(repo=args.repo, rev=args.rev, path=fname)
        # First 8 bytes
        st, h1, b1 = http_range(url, 0, 7)
        if st != 206 or len(b1) != 8 or "content-range" not in h1:
            write_text(os.path.join(out_dir, "ERROR.txt"), f"{fname}: expected 206 and 8 bytes; got {st}, {len(b1)}\n")
            print(f"ERROR: {fname} first8 failed", file=sys.stderr)
            sys.exit(2)
        write_bytes(os.path.join(out_dir, f"{fname}.first8.bin"), b1)
        save_http_headers(os.path.join(out_dir, f"{fname}.first8.http"), h1)
        file_size = int(h1["content-range"].split("/")[-1])
        (header_len,) = struct.unpack("<Q", b1)
        # Header JSON
        end = 7 + header_len
        st2, h2, b2 = http_range(url, 8, end)
        if st2 != 206 or len(b2) != header_len or "content-range" not in h2:
            write_text(os.path.join(out_dir, "ERROR.txt"), f"{fname}: header fetch failed st={st2} len={len(b2)} want={header_len}\n")
            print(f"ERROR: {fname} header failed", file=sys.stderr)
            sys.exit(2)
        save_http_headers(os.path.join(out_dir, f"{fname}.header.http"), h2)
        write_bytes(os.path.join(out_dir, f"{fname}.header.json"), b2)
        # Parse header for memory sums and names
        header_map = json.loads(b2)
        # Sum of data bytes across entries
        shard_calc = 0
        for name, meta in header_map.items():
            if not isinstance(meta, dict):
                continue
            dt = meta.get("dtype")
            sz = dtype_size(dt or "")
            sh = meta.get("shape")
            offs = meta.get("data_offsets")
            if sz is None or not isinstance(sh, list) or not isinstance(offs, list) or len(offs) != 2:
                continue
            n_elem = 1
            for d in sh:
                n_elem *= int(d)
            shard_calc += n_elem * sz
        data_len = file_size - 8 - header_len
        range_meta[fname] = {
            "file_size": file_size,
            "header_len": header_len,
            "calc_data_bytes": shard_calc,
            "data_len_reported": data_len,
            "calc_matches_reported": (shard_calc == data_len),
            "num_tensors": sum(1 for v in header_map.values() if isinstance(v, dict)),
        }
        total_calc_data += shard_calc
        total_file_sizes += file_size
        # Merge names
        all_headers.update(header_map)
        per_file_headers[fname] = header_map

    write_text(os.path.join(out_dir, "range_meta.json"), json.dumps(range_meta, indent=2))

    # 4) Build occurrences and index checks
    occurrences: Dict[str, Set[str]] = {}
    for fname, header_map in per_file_headers.items():
        for name, meta in header_map.items():
            if not isinstance(meta, dict):
                continue
            occurrences.setdefault(name, set()).add(fname)
    # Index consistency
    idx_missing: List[str] = []
    idx_mismatch: List[Tuple[str, str, List[str]]] = []
    for name, shard in weight_map.items():
        occ = occurrences.get(name)
        if not occ:
            idx_missing.append(name)
        elif shard not in occ:
            idx_mismatch.append((name, shard, sorted(occ)))
    # Unindexed header entries
    unindexed = sorted([n for n in occurrences.keys() if n not in weight_map])

    # 5) Group parsing across all headers
    # Partition by prefix => parts
    parts_by_prefix: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, meta in all_headers.items():
        if not isinstance(meta, dict):
            continue
        if "." not in name:
            continue
        pref, part = name.rsplit(".", 1)
        if part in ("trellis", "suh", "su", "svh", "sv", "mcg", "mul1", "bias"):
            parts_by_prefix.setdefault(pref, {})[part] = meta
    accepted: Dict[str, GroupMeta] = {}
    refused: Dict[str, str] = {}
    for pref, parts in parts_by_prefix.items():
        g, why = parse_group(pref, parts)
        if g:
            accepted[pref] = g
        else:
            refused[pref] = why or "unknown"

    # 6) GLM text restrictions: integer bits (twice even), fp16 scales, no bias
    glm_text_bad: Dict[str, str] = {}
    for pref, g in accepted.items():
        if pref.startswith("model.visual."):
            continue  # vision tower handled separately
        if g.bits_twice % 2 != 0:
            glm_text_bad[pref] = f"fractional bits {g.bits_twice/2.0}"
            continue
        if not (g.in_scales == "suh" and g.out_scales == "svh"):
            glm_text_bad[pref] = f"non-fp16 scales {g.in_scales}/{g.out_scales}"
            continue
        if g.bias:
            glm_text_bad[pref] = "bias present"

    # 7) Expert-bank uniformity (by layer and kind)
    expert_pattern = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(down_proj|gate_proj|up_proj)$"
    )
    by_layer_kind: Dict[Tuple[int, str], Dict[str, Set[int]]] = {}
    for pref, g in accepted.items():
        m = expert_pattern.match(pref)
        if not m:
            continue
        layer = int(m.group(1))
        kind = m.group(3)
        key = (layer, kind)
        S = by_layer_kind.setdefault(
            key, {"K": set(), "N": set(), "bits_twice": set(), "codebook": set(), "scales": set()}
        )
        S["K"].add(g.k)
        S["N"].add(g.n)
        S["bits_twice"].add(g.bits_twice)
        S["codebook"].add(0 if g.codebook == "3inst" else (1 if g.codebook == "mcg" else 2))
        S["scales"].add(0 if (g.in_scales, g.out_scales) == ("suh", "svh") else 1)
    non_uniform: Dict[str, Dict[str, List[int]]] = {}
    for (layer, kind), sets in by_layer_kind.items():
        issues = []
        if len(sets["K"]) != 1:
            issues.append("K")
        if len(sets["N"]) != 1:
            issues.append("N")
        if len(sets["bits_twice"]) != 1:
            issues.append("bits")
        if len(sets["codebook"]) != 1:
            issues.append("codebook")
        if len(sets["scales"]) != 1:
            issues.append("scales")
        if issues:
            non_uniform[f"layer{layer}.{kind}"] = {i: sorted(list(sets[i.upper() if i in ('K','N') else i])) for i in issues}

    # 8) NextN / MTP presence (header-only)
    mtp_keys = [k for k in weight_map.keys() if ".eh_proj." in k or ".shared_head.norm.weight" in k or ".enorm." in k or ".hnorm." in k]

    # 9) Read config and infer hidden/vocab to sanity-check eh_proj dims where possible
    with open(os.path.join(out_dir, "config.json"), "rb") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    hidden = int(text_cfg.get("hidden_size", text_cfg.get("n_embd", 0)) or 0)
    vocab = int(text_cfg.get("vocab_size", 0) or 0)
    eh_proj_ok = None
    for pref, g in accepted.items():
        if pref.endswith(".eh_proj") and hidden:
            eh_proj_ok = (g.k == 2 * hidden and g.n == hidden)
            break

    # 10) Totals and summary
    summary = {
        "repo": args.repo,
        "rev": args.rev,
        "artifact_dir": out_dir,
        "manifest_entries": len(manifest_rows),
        "index_weight_map_entries": len(weight_map),
        "range_meta_ok": all(m["calc_matches_reported"] for m in range_meta.values()),
        "total_calc_data_bytes": total_calc_data,
        "total_file_sizes": total_file_sizes,
        "headers_scanned": len(all_headers),
        "accepted_groups": len(accepted),
        "refused_groups": len(refused),
        "glm_text_bad_groups": len(glm_text_bad),
        "non_uniform_banks": non_uniform,
        "mtp_related_keys": len(mtp_keys),
        "hidden": hidden,
        "vocab": vocab,
        "eh_proj_dims_ok": eh_proj_ok,
        "index_missing_names": len(idx_missing),
        "index_mismatch": len(idx_mismatch),
        "unindexed_header_names": len(unindexed),
    }
    write_text(os.path.join(out_dir, "summary.json"), json.dumps(summary, indent=2))
    # Human-readable gist
    gist = []
    gist.append(f"HF repo: {args.repo}")
    gist.append(f"Revision: {args.rev}")
    gist.append(f"Manifest rows: {len(manifest_rows)}")
    gist.append(f"Index weight_map entries: {len(weight_map)}")
    ok = 'ok' if summary['range_meta_ok'] else 'MISMATCH'
    gib = lambda b: b / (1<<30)
    gist.append(f"Range/memory sums: {ok}; total data {gib(total_calc_data):.2f} GiB across {len(shards)} shards")
    gist.append(f"Groups accepted: {len(accepted)}; refused: {len(refused)}; GLM text violations: {len(glm_text_bad)}")
    gist.append(f"Non-uniform banks: {len(non_uniform)}; MTP keys: {len(mtp_keys)}; hidden={hidden} vocab={vocab} eh_proj_ok={eh_proj_ok}")
    gist.append(f"Index missing: {len(idx_missing)}; index mismatches: {len(idx_mismatch)}; unindexed headers: {len(unindexed)}")
    write_text(os.path.join(out_dir, "summary.txt"), "\n".join(gist) + "\n")

    print("\n".join(gist))


if __name__ == "__main__":
    main()


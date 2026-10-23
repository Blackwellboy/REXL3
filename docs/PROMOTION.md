# Promotion: dev approval to production

Production (`REXL3/REXL3`) holds only runtime trees that a dev-side approval sealed. Nothing is
copied by hand, and every hop is a hash comparison.

## The three artifacts

| Artifact | Made where | What it proves |
| --- | --- | --- |
| `approve-<runtime-id>-<n>` (dev release) | the GB10 host that has the pack and the binary | the commit, the template, the archive hash and the gate results |
| `approval.json` | inside that release | `rexl3.profile-approval.v1`: commit + tree, template version + digest, profile digest, archive sha256/bytes/files, gates, approver, timestamp |
| `runtimes/<runtime-id>/v<n>/` (prod tree) | `promote.yml` in prod | the same files, byte for byte, minus `bin/` and `release.lock.json` |

The runnable bundle is always the release asset, never a git tree: it carries `bin/rexl3-serve`
and `bin/rexl3-cluster`, which a CPU CI runner cannot build. A bare clone of production has
everything needed to verify a tree and nothing needed to serve.

## Why the seal runs on the host, not in CI

`bin/rexl3-serve` is a CUDA build. A hosted runner cannot produce it, and a self-hosted runner on a
private repository is a liability we are not adding for this. So the seal is a script an operator
runs where the binary was qualified:

```bash
PROFILE_DIGEST=sha256:<engine /health value at phase=ready> \
tools/approve_release.sh <runtime-id> approve-<runtime-id>-<n> \
    <native-bin-dir> <release-input.json> <gates.json>
```

It refuses, in order: a missing or malformed `PROFILE_DIGEST`, modified tracked files, uncommitted
files under `runtimes/ LICENSE NOTICE THIRD_PARTY_NOTICES.md sbom/`, a missing profile, a missing
binary, gates that do not cover `doctor`/`exactness`/`capability`, and a `release-input.json` whose
`production_commit` is not `HEAD`. Then it regenerates the tree, builds the bundle, reproduces it
with `--check`, writes the approval and publishes the release.

## Promotion

```bash
python3 tools/promote_runtime.py --runtime-id <id> --approve-tag approve-<id>-<n> --version <n>
```

`promote.yml` downloads both assets, checks the archive's sha256 and size against the approval,
refuses a template digest that is not in `runtimes/templates/APPROVED.json`, extracts the tree
without `bin/`, opens `promote/<id>-v<n>` and enables auto-merge. When `verify` is green the PR
merges and `release.yml` tags `prod/<id>/v<n>` and attaches the same archive.

## Verification in production

`verify.yml` runs on every pull request and every push to `main`:

- `tools/verify_runtime_tree.py` walks `runtimes/*/v*/`, requires `approval.json` and
  `source.manifest.json`, refuses an unapproved template version or digest, and compares every file
  against the manifest — naming the path and both hashes on a mismatch, and refusing extra or
  missing files.
- The verifier's own tests run beside it.

Cost is seconds: hashing a runtime tree is a few megabytes. There is no Rust build and no GPU.

## Tags

| Tag | Meaning |
| --- | --- |
| `approve-<runtime-id>-<n>` | dev release; `<n>` and the production `v<n>` are the same integer |
| `prod/<runtime-id>/v<n>` | production release; same archive bytes as the dev release |

Two people promoting at once can collide on the number. `release.yml` refuses to overwrite an
existing tag; pick the next integer and re-run.

## Withdrawing a bad pack

Production history is append-only. If a tree turns out wrong: revert the promotion commit, tag the
revert (`prod/<id>/v<n>-withdrawn`), and edit `runtimes/<id>/LATEST` to the previous version. Do not
force-push `main` — the promotion PR is the audit trail.

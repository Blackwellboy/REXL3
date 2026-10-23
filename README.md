# REXL3 production runtime packs

This repository holds **approved runtime packs** for REXL3 serving: the serve profile, the launch
and contract scripts, the model and dependency locks, and the approval document that binds each
tree to the commit and template it was built from. Nothing here is hand-edited after promotion.

The runnable bundle is a **release asset**, not a git tree. It carries the CUDA binaries
(`bin/rexl3-serve`, `bin/rexl3-cluster`), which a CPU CI runner cannot build. To run a pack, download
its release:

```bash
gh release download prod/dsv41-flash-1017-sage-1.62/v1 -R REXL3/REXL3
tar -xzf dsv41-flash-1017-sage-1.62-1.tar.gz -C ~/dsv41 && cd ~/dsv41
./launch.sh install-deps && ./launch.sh doctor && ./launch.sh setup && ./launch.sh start
```

A bare clone of this repository verifies provenance; it does not serve. Every tree is checked on
every push: `tools/verify_runtime_tree.py` recomputes each file's sha256 against
`source.manifest.json`, refuses an unapproved template digest, and refuses extra or missing files.

## Layout

```
runtimes/
  templates/APPROVED.json        # template versions and digests production will accept
  <runtime-id>/
    LATEST                       # the promoted version, e.g. v1
    v<N>/                        # one promoted tree
      approval.json              # rexl3.profile-approval.v1: commit, template, archive, gates
      source.manifest.json       # every file's sha256, bytes and mode
      profile.json, model.lock.json, dependencies.lock.json, requirements-runtime.lock,
      KNOWN_ISSUES.md, handoff.json, LICENSE, NOTICE, README.md, launch.sh, scripts/, docs/
```

## How a pack gets here

1. On the GB10 host: `tools/approve_release.sh` in `REXL3-dev` seals the current commit, builds the
   bundle, reproduces it byte-for-byte, writes `approval.json` and publishes the dev release
   `approve-<runtime-id>-<n>`.
2. `python3 tools/promote_runtime.py --runtime-id … --approve-tag … --version …` dispatches
   `promote.yml`, which re-verifies the archive and template, opens a pull request and enables
   auto-merge.
3. Once `verify` is green the promotion PR is rebased by the `merge` job; the operator's command then
   publishes `prod/<runtime-id>/v<n>` with the same archive bytes.

The full rules, including withdrawal, are in `docs/PROMOTION.md`.

## Licence

AGPL-3.0-only (`LICENSE`, `NOTICE`) covers everything here that the REXL3 authors wrote. Model
weights are not in this repository: each pack's `model.lock.json` pins the upstream repository and
revision, and that repository's own licence governs the weights.

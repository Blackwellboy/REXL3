#!/usr/bin/env python3
"""Verify production runtime trees against the approvals that sealed them.

Production holds only approved trees. For every ``runtimes/<id>/v<N>/`` this checks that
``approval.json`` is present and well formed, that the template version it names is on the
repository's approved list, and that every file matches ``source.manifest.json`` byte for byte —
no extra file, no missing file, no edited file. Exit 1 with a named reason per problem.

CLI: python3 tools/verify_runtime_tree.py --root .
"""
import argparse
import hashlib
import json
import os
import sys

SCHEMA = "rexl3.profile-approval.v1"
IGNORED = {"approval.json", "source.manifest.json", "LATEST"}
REQUIRED_GATES = ("doctor", "exactness", "capability")


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_tree(tree, approved_templates):
    """Every problem with ``tree``; an empty list means it matches its approval exactly."""
    problems = []
    approval_path = os.path.join(tree, "approval.json")
    manifest_path = os.path.join(tree, "source.manifest.json")
    for path in (approval_path, manifest_path):
        if not os.path.isfile(path):
            return [f"{tree}: missing {os.path.basename(path)}"]
    try:
        with open(approval_path, encoding="utf-8") as handle:
            approval = json.load(handle)
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        return [f"{tree}: unreadable approval or manifest: {error}"]
    if approval.get("schema") != SCHEMA:
        problems.append(f"{tree}: approval schema is {approval.get('schema')}, expected {SCHEMA}")
    missing_gates = [g for g in REQUIRED_GATES if not (approval.get("gates") or {}).get(g)]
    if missing_gates:
        problems.append(f"{tree}: approval is missing gate results: {', '.join(missing_gates)}")
    bundle = approval.get("bundle") or {}
    for field in ("archive", "sha256", "bytes", "files"):
        if not bundle.get(field):
            problems.append(f"{tree}: approval bundle.{field} is missing")
    template = approval.get("template") or {}
    expected = approved_templates.get(template.get("version"))
    if expected is None:
        problems.append(f"{tree}: template version {template.get('version')} is not approved")
    elif expected != template.get("digest"):
        problems.append(f"{tree}: template digest {template.get('digest')} is not the approved {expected}")
    manifest_template = (manifest.get("template") or {})
    if manifest_template.get("digest") != template.get("digest"):
        problems.append(
            f"{tree}: source manifest names template {manifest_template.get('digest')}, "
            f"the approval names {template.get('digest')}"
        )
    files = manifest.get("files") or {}
    if not files:
        problems.append(f"{tree}: source manifest lists no files")
    on_disk = {}
    for base, dirs, names in os.walk(tree):
        dirs[:] = sorted(dirs)
        for name in sorted(names):
            rel = os.path.relpath(os.path.join(base, name), tree).replace(os.sep, "/")
            if rel in IGNORED:
                continue
            on_disk[rel] = sha256_file(os.path.join(base, name))
    for rel, record in sorted(files.items()):
        if rel not in on_disk:
            problems.append(f"{tree}: {rel} is listed but missing")
        elif on_disk[rel] != record.get("sha256"):
            problems.append(f"{tree}: {rel} sha256 {on_disk[rel]} != approved {record.get('sha256')}")
    for rel in sorted(on_disk):
        if rel not in files:
            problems.append(f"{tree}: {rel} is not in the approved manifest")
    return problems


def discover(root):
    """Every runtimes/<id>/v<N>/ tree under root."""
    runtimes = os.path.join(root, "runtimes")
    trees = []
    if not os.path.isdir(runtimes):
        return trees
    for name in sorted(os.listdir(runtimes)):
        base = os.path.join(runtimes, name)
        if name in ("templates", "template-v1") or not os.path.isdir(base):
            continue
        for version in sorted(os.listdir(base)):
            candidate = os.path.join(base, version)
            if version.startswith("v") and os.path.isdir(candidate):
                trees.append(candidate)
    return trees


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--approved", default=None,
                        help="approved templates JSON (default runtimes/templates/APPROVED.json)")
    args = parser.parse_args(argv)
    approved_path = args.approved or os.path.join(args.root, "runtimes", "templates", "APPROVED.json")
    if not os.path.isfile(approved_path):
        print(f"no approved template list: {approved_path}", file=sys.stderr)
        return 2
    with open(approved_path, encoding="utf-8") as handle:
        approved = json.load(handle)["templates"]
    trees = discover(args.root)
    if not trees:
        print("no runtime trees found")
        return 0
    problems = []
    for tree in trees:
        problems.extend(verify_tree(tree, approved))
    for problem in problems:
        print(problem)
    print(f"{len(trees)} tree(s), {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for the production tree verifier.

Run: python3 -m unittest tools.test_verify_runtime_tree
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_runtime_tree as v

TEMPLATE_DIGEST = "sha256:" + "c" * 64


def write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.tree = os.path.join(self.root, "runtimes", "demo", "v1")
        files = {
            "profile.json": write(os.path.join(self.tree, "profile.json"), b"{}"),
            "LICENSE": write(os.path.join(self.tree, "LICENSE"), b"agpl"),
        }
        write(os.path.join(self.tree, "source.manifest.json"),
              json.dumps({"files": files, "template": {"version": "v1", "digest": TEMPLATE_DIGEST}}).encode())
        write(os.path.join(self.tree, "approval.json"), json.dumps({
            "schema": "rexl3.profile-approval.v1", "runtime_id": "demo",
            "source_commit": "a" * 40, "source_tree": "b" * 40,
            "template": {"version": "v1", "digest": TEMPLATE_DIGEST},
            "profile_digest": "sha256:" + "d" * 64,
            "bundle": {"archive": "a.tar.gz", "sha256": "e" * 64, "bytes": 1, "files": 2},
            "gates": {"doctor": "PASS", "exactness": "15/15", "capability": "8/8"},
            "approved_by": "r0b0tlab", "approved_at": "2026-10-23T00:00:00Z"}).encode())
        self.approved = {"v1": TEMPLATE_DIGEST}

    def test_clean_tree_passes(self):
        self.assertEqual(v.verify_tree(self.tree, self.approved), [])

    def test_tampered_file_is_named_with_both_hashes(self):
        write(os.path.join(self.tree, "LICENSE"), b"mit")
        problems = v.verify_tree(self.tree, self.approved)
        self.assertTrue(any("LICENSE" in p and "approved" in p for p in problems), problems)

    def test_extra_file_is_refused(self):
        write(os.path.join(self.tree, "extra.txt"), b"hi")
        self.assertTrue(any("extra.txt" in p for p in v.verify_tree(self.tree, self.approved)))

    def test_missing_file_is_refused(self):
        os.unlink(os.path.join(self.tree, "LICENSE"))
        self.assertTrue(any("LICENSE" in p and "missing" in p for p in v.verify_tree(self.tree, self.approved)))

    def test_unapproved_template_is_refused(self):
        problems = v.verify_tree(self.tree, {"v1": "sha256:" + "f" * 64})
        self.assertTrue(any("template" in p for p in problems), problems)

    def test_unknown_template_version_is_refused(self):
        problems = v.verify_tree(self.tree, {})
        self.assertTrue(any("not approved" in p for p in problems), problems)

    def test_missing_approval_is_refused(self):
        os.unlink(os.path.join(self.tree, "approval.json"))
        self.assertTrue(any("approval.json" in p for p in v.verify_tree(self.tree, self.approved)))


if __name__ == "__main__":
    unittest.main()

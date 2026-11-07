import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[0] / "dev" / "serve_glm53_2p48.py"


class TestDevServeGLM532p48(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pack = Path(self.tmp.name) / "pack"
        self.pack.mkdir(parents=True, exist_ok=True)
        # minimal file presence checks used by the script
        for name in ("config.json", "generation_config.json", "tokenizer.json", "model.safetensors.index.json"):
            (self.pack / name).write_text("{}", encoding="utf-8")
        # create a fake rexl3-serve that prints the version keys the script expects
        self.serve = Path(self.tmp.name) / "rexl3-serve"
        self.serve.write_text("#!/usr/bin/env bash\necho 'version 0.0'\necho 'commit abcdef0'\necho 'system_fingerprint fp'\necho 'nvcc 13.0'\necho 'cuda_arch sm_121'\necho 'kernel_hash deadbeef'\necho 'cudart 13.0.96'\necho 'cuda_driver 580.178.04'\necho 'cublas 13.1.1.3'\n", encoding="utf-8")
        os.chmod(self.serve, 0o755)

    def test_print_plan_json(self):
        # pick a high port likely to be free
        port = 18955
        cmd = [sys.executable, str(SCRIPT),
               "--serve-bin", str(self.serve),
               "--model-dir", str(self.pack),
               "--host", "127.0.0.1",
               "--port", str(port),
               "--slots", "1",
               "--check-port"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        plan = json.loads(proc.stdout)
        self.assertEqual(plan["host"], "127.0.0.1")
        self.assertEqual(plan["port"], port)
        self.assertEqual(plan["slots"], 1)
        argv = plan["argv"]
        self.assertIn(str(self.serve), argv)
        self.assertIn("--model", argv)
        self.assertIn("--host", argv)
        self.assertIn("--port", argv)
        self.assertIn("--slots", argv)
        # Must not include any override that weakens gates
        for forbidden in ("--allow-unvalidated-pack", "--prefix-cache", "--max-model-len"):
            self.assertNotIn(forbidden, argv)
        # Native version string should be present
        self.assertIn("cuda_driver", plan["native_version"])

    def test_port_check_reports_occupied(self):
        # Bind a local socket to simulate occupation
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        try:
            cmd = [sys.executable, str(SCRIPT),
                   "--serve-bin", str(self.serve),
                   "--model-dir", str(self.pack),
                   "--host", "127.0.0.1",
                   "--port", str(port),
                   "--slots", "1",
                   "--check-port"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            plan = json.loads(proc.stdout)
            self.assertEqual(plan["port_status"], "occupied")
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()


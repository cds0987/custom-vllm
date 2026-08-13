"""Structure rules, enforced (transformers' make check-repo, miniature).

Rules from models/_template/MANIFEST.md:
1. No cross-model imports (models/a must not import models/b).
2. Root utils/ must not import from models/ (dependency points one way).
3. Every load/hardware variant file and engine adapter.py carries a
   module-level ADAPTER = {...} literal with axis+variant keys.
4. models/auto/registry.py scans without importing — must succeed and
   report qwen3_5 with all three axes.

Run: python tests/test_structure.py
"""

import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"


def model_dirs():
    return [d for d in MODELS.iterdir()
            if d.is_dir() and d.name not in ("auto", "_template", "__pycache__")]


def py_files(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def imports_of(py: Path):
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except SyntaxError as e:
        raise AssertionError(f"{py}: syntax error {e}")
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


class TestStructure(unittest.TestCase):
    def test_no_cross_model_imports(self):
        models = [d.name for d in model_dirs()]
        for d in model_dirs():
            others = [m for m in models if m != d.name]
            for py in py_files(d):
                for imp in imports_of(py):
                    for other in others:
                        self.assertFalse(
                            imp.startswith(f"models.{other}") or imp == other,
                            f"{py} imports sibling model '{other}' — copy with "
                            f"'# Copied from' instead (rule 2, _template)")

    def test_root_utils_never_imports_models(self):
        for py in py_files(REPO / "utils"):
            for imp in imports_of(py):
                self.assertFalse(imp.startswith("models"),
                                 f"{py} imports models/* — utils must stay model-neutral")

    def test_adapter_metadata_present(self):
        missing = []
        for d in model_dirs():
            candidates = []
            for axis in ("load", "hardware"):
                if (d / axis).is_dir():
                    candidates += [p for p in (d / axis).glob("*.py")
                                   if not p.name.startswith("_")
                                   and p.name != "register.py"
                                   and not p.name.startswith("legacy_")
                                   and not p.name.startswith("quantize_")
                                   and not p.name.startswith("graft_lm_head")]
            if (d / "engine").is_dir():
                candidates += list((d / "engine").glob("*/adapter.py"))
            for py in candidates:
                src = py.read_text(encoding="utf-8")
                if not re.search(r"^ADAPTER\s*=\s*\{", src, re.M):
                    missing.append(str(py))
        self.assertEqual(missing, [], f"files without ADAPTER = {{...}}: {missing}")

    def test_registry_scans_clean(self):
        r = subprocess.run([sys.executable, str(REPO / "register.py")],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, f"registry failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn('"qwen3_5"', r.stdout)
        for axis in ("engine", "load", "hardware"):
            self.assertIn(f'"{axis}"', r.stdout, f"registry misses axis {axis}")
        self.assertNotIn("_error", r.stdout, "a register.py/ADAPTER literal failed to parse")

    def test_every_section_has_register(self):
        # recursive-registry rule: every product folder ships register.py;
        # a folder without one is invisible (that must be a CHOICE, not a miss)
        must_register = ["models", "models/qwen3_5", "models/qwen3_5/engine",
                         "models/qwen3_5/load", "models/qwen3_5/hardware",
                         "bench", "bench/workload", "sdk", "loading", "logging",
                         "utils", "tests"]
        missing = [d for d in must_register if not (REPO / d / "register.py").exists()]
        self.assertEqual(missing, [], f"folders without register.py: {missing}")

    def test_patches_live_beside_what_they_patch(self):
        # engine patches must not sit at model level anymore
        for d in model_dirs():
            stray = list(d.glob("patches/patch_vllm_*.py")) + list(d.glob("patches/patch_gguf_*.py"))
            self.assertEqual(stray, [], f"engine-specific patches at model level: {stray}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

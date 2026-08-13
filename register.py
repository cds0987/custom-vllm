#!/usr/bin/env python3
"""Recursive registry — the ONE entrypoint that registers everything.

Every folder that wants to exist in the product ships its own `register.py`
declaring a module-level REGISTER = {...} literal. This root walker descends
ONLY into subfolders that carry one — a folder without register.py is
invisible by design (that's the control knob: implementation details like
engine/vllm/patches/ stay out of the product surface).

Inside a registered folder, any *.py file with a module-level ADAPTER = {...}
literal is picked up as a leaf variant (engine adapters, load paths, hardware
envelopes). Both literals are read with ast — never imported — so heavy
imports in adapters cost nothing here.

    python register.py            # full support tree as JSON
    python register.py --flat     # flat "path: kind - description" listing

Adding support = drop a folder with register.py (+ files with ADAPTER).
Nothing existing is ever edited. (transformers' models/auto, made recursive.)
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REGISTER = {
    "name": "custom-vllm",
    "kind": "root",
    "description": "Adapter hoa LLM serving cho business: models (kien truc da thuan hoa) "
                   "+ sdk/loading/logging (tang ngoai) + bench (do dac trung lap)",
}


def _literal(py_file: Path, var: str) -> dict:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == var:
                        return ast.literal_eval(node.value)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    return {}


def collect(folder: Path) -> dict:
    node = dict(_literal(folder / "register.py", "REGISTER"))
    adapters = {}
    for py in sorted(folder.glob("*.py")):
        if py.name in ("register.py",) or py.name.startswith("_"):
            continue
        a = _literal(py, "ADAPTER")
        if a:
            adapters[py.stem] = a
    if adapters:
        node["adapters"] = adapters
    children = {}
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if (sub / "register.py").exists():
            children[sub.name] = collect(sub)
        elif (sub / "adapter.py").exists():  # variant folder: adapter.py doubles as its register
            entry = dict(_literal(sub / "adapter.py", "ADAPTER"))
            n_patches = len(list((sub / "patches").glob("*.py"))) if (sub / "patches").is_dir() else 0
            if n_patches:
                entry["patches"] = n_patches
            children[sub.name] = entry
    if children:
        children_key = "children"
        node[children_key] = children
    return node


def flat(node: dict, path: str = "") -> list:
    rows = [f"{path or '.'}: [{node.get('kind', '?')}] {node.get('description', node.get('name', ''))}"]
    for name, a in (node.get("adapters") or {}).items():
        rows.append(f"{path}/{name}: [{a.get('axis', 'adapter')}] {a.get('variant', name)}")
    for name, child in (node.get("children") or {}).items():
        if "children" in child or "adapters" in child or "kind" in child:
            rows += flat(child, f"{path}/{name}")
        else:
            rows.append(f"{path}/{name}: [variant] {child.get('variant', name)}")
    return rows


def main() -> int:
    tree = collect(ROOT)
    if "--flat" in sys.argv:
        print("\n".join(flat(tree)))
    else:
        print(json.dumps(tree, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

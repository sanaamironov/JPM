from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(out_path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def append_csv(out_path: Path, row: Dict[str, Any]) -> None:
    import csv
    ensure_dir(out_path.parent)
    exists = out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
"""
format_section4_tables.py

Post-processing utility: combine per-cell outputs from replicate_section4.py into
paper-friendly tables.

Reads files produced by the canonical replication runner:
  jpm_q3.lu25.experiments.replicate_section4

Expected input directory structure
----------------------------------
Given an input root (e.g., results/part2/lu25_section4), we expect subfolders like:
  <in>/DGP1_T25_J15/
  <in>/DGP2_T25_J15/
  ...

Within each subfolder we expect (produced by replicate_section4.py):
  - paper_table_like.csv   (required)
  - summary.csv            (optional; long format)

Outputs
-------
Written under --out (default: <in>):
  - paper_table_like_combined.csv  : stacked rows from all cells
  - paper_table_wide.csv           : wide pivot table (one row per cell+method+row)
  - summary_long_combined.csv      : stacked summary.csv rows (if found)

Usage
-----
Combine everything found under the input root:
  python -m jpm_q3.lu25.experiments.format_section4_tables --in results/part2/lu25_section4

Combine a specific grid subset (only those cells):
  python -m jpm_q3.lu25.experiments.format_section4_tables \
    --in results/part2/lu25_section4 \
    --grid DGP1:25:15,DGP2:25:15

Notes
-----
- If some cells are missing, we will warn and continue.
- The wide output is designed to be easy to copy into your report.

Build:
    python -m jpm_q3.lu25.experiments.format_section4_tables --in results/part2/lu25_section4
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GridPoint:
    dgp: str
    T: int
    J: int

    @property
    def cell(self) -> str:
        return f"{self.dgp}_T{self.T}_J{self.J}"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Format/merge Lu(25) Section 4 outputs into paper tables.")
    p.add_argument(
        "--in",
        dest="in_dir",
        type=str,
        required=True,
        help="Input root directory produced by replicate_section4.py (contains per-cell folders).",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        type=str,
        default="",
        help="Output directory. Default: same as --in.",
    )
    p.add_argument(
        "--grid",
        type=str,
        default="",
        help="Optional subset grid: DGP1:25:15,DGP2:25:15. If empty, auto-detect folders under --in.",
    )
    return p.parse_args(argv)


def _parse_grid_spec(spec: str) -> List[GridPoint]:
    pts: List[GridPoint] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad --grid token '{token}'. Expected format DGP#:T:J")
        dgp, T, J = parts[0].strip().upper(), int(parts[1]), int(parts[2])
        pts.append(GridPoint(dgp=dgp, T=T, J=J))
    if not pts:
        raise ValueError("Parsed --grid but got empty list.")
    return pts


def _detect_grid_from_folders(in_root: Path) -> List[GridPoint]:
    """
    Detect grid points from folder names like DGP2_T25_J15.
    """
    pts: List[GridPoint] = []
    for child in sorted(in_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        # expected: <DGP>_T<T>_J<J>
        parts = name.split("_")
        if len(parts) != 3:
            continue
        dgp = parts[0].upper()
        if not dgp.startswith("DGP"):
            continue
        if not parts[1].startswith("T") or not parts[2].startswith("J"):
            continue
        try:
            T = int(parts[1][1:])
            J = int(parts[2][1:])
        except Exception:
            continue
        pts.append(GridPoint(dgp=dgp, T=T, J=J))
    return pts


# -----------------------------------------------------------------------------
# CSV utilities
# -----------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def combine_paper_table_like(in_root: Path, grid: List[GridPoint]) -> Tuple[List[Dict[str, object]], List[str]]:
    """
    Reads each cell's paper_table_like.csv and returns combined rows.

    replicate_section4.py writes columns:
      DGP,T,J,Method,Row,Int,beta_p,beta_w,sigma,xi_mean_abs_error,xi_sd_error,Prob_signal,Prob_noise,FailRate

    We'll keep those, and also add:
      cell (DGP_T#_J#) and source_path
    """
    combined: List[Dict[str, object]] = []
    warnings: List[str] = []

    for gp in grid:
        cell_dir = in_root / gp.cell
        src = cell_dir / "paper_table_like.csv"
        if not src.exists():
            warnings.append(f"missing {src}")
            continue

        rows = _read_csv_rows(src)
        for r in rows:
            out = dict(r)
            out["cell"] = gp.cell
            out["source_path"] = str(src)
            combined.append(out)

    return combined, warnings


def combine_summary_long(in_root: Path, grid: List[GridPoint]) -> Tuple[List[Dict[str, object]], List[str]]:
    """
    Combines per-cell summary.csv (long format) into one file, if present.

    replicate_section4.py writes:
      dgp,T,J,method,metric,value
    We'll add:
      cell, source_path
    """
    combined: List[Dict[str, object]] = []
    warnings: List[str] = []

    for gp in grid:
        cell_dir = in_root / gp.cell
        src = cell_dir / "summary.csv"
        if not src.exists():
            continue  # optional
        rows = _read_csv_rows(src)
        for r in rows:
            out = dict(r)
            out["cell"] = gp.cell
            out["source_path"] = str(src)
            combined.append(out)

    if not combined:
        warnings.append("no summary.csv files found (this is OK).")
    return combined, warnings


def pivot_wide_paper_table(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Pivot paper_table_like_combined into a wide table:

    Key:
      cell, DGP, T, J, Method, Row
    Columns:
      Int, beta_p, beta_w, sigma, xi_mean_abs_error, xi_sd_error, Prob_signal, Prob_noise, FailRate

    This is mostly already "wide" in your per-cell files; this function mainly:
    - ensures types are consistent
    - ensures stable ordering
    - removes redundant columns like source_path
    """
    out_rows: List[Dict[str, object]] = []

    keep_cols = [
        "cell",
        "DGP",
        "T",
        "J",
        "Method",
        "Row",
        "Int",
        "beta_p",
        "beta_w",
        "sigma",
        "xi_mean_abs_error",
        "xi_sd_error",
        "Prob_signal",
        "Prob_noise",
        "FailRate",
    ]

    for r in rows:
        out: Dict[str, object] = {}
        # normalize and cast some fields
        out["cell"] = r.get("cell", "")
        out["DGP"] = r.get("DGP", "")
        out["T"] = r.get("T", "")
        out["J"] = r.get("J", "")
        out["Method"] = r.get("Method", "")
        out["Row"] = r.get("Row", "")

        # numeric-ish columns as strings are fine, but we can normalize "nan" -> empty
        for k in ["Int", "beta_p", "beta_w", "sigma", "xi_mean_abs_error", "xi_sd_error", "Prob_signal", "Prob_noise", "FailRate"]:
            v = r.get(k, "")
            f = _to_float(v)
            out[k] = "" if f is None else f

        out_rows.append(out)

    # stable sort
    def _sort_key(rr: Dict[str, object]) -> Tuple:
        return (
            str(rr.get("DGP", "")),
            int(rr.get("T") or 0),
            int(rr.get("J") or 0),
            str(rr.get("Method", "")),
            str(rr.get("Row", "")),
        )

    out_rows.sort(key=_sort_key)
    return out_rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    in_root = Path(args.in_dir)
    if not in_root.exists():
        raise FileNotFoundError(f"--in directory does not exist: {in_root}")

    out_root = Path(args.out_dir) if args.out_dir.strip() else in_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.grid.strip():
        grid = _parse_grid_spec(args.grid)
    else:
        grid = _detect_grid_from_folders(in_root)

    if not grid:
        raise ValueError(
            f"No grid cells found under {in_root}. "
            "Either pass --grid explicitly (e.g., DGP1:25:15) or ensure per-cell folders exist."
        )

    print(f"[format] in:  {in_root}")
    print(f"[format] out: {out_root}")
    print(f"[format] cells: {len(grid)}")
    for gp in grid:
        print(f"  - {gp.cell}")

    # Combine paper_table_like.csv
    combined_rows, warn1 = combine_paper_table_like(in_root, grid)
    if warn1:
        for w in warn1:
            print(f"[format][WARN] {w}")

    if not combined_rows:
        raise ValueError("No paper_table_like.csv rows found. Did replicate_section4.py finish successfully?")

    combined_path = out_root / "paper_table_like_combined.csv"
    # Fieldnames: preserve the canonical columns + a couple extras
    combined_fields = [
        "cell",
        "source_path",
        "DGP",
        "T",
        "J",
        "Method",
        "Row",
        "Int",
        "beta_p",
        "beta_w",
        "sigma",
        "xi_mean_abs_error",
        "xi_sd_error",
        "Prob_signal",
        "Prob_noise",
        "FailRate",
    ]
    _write_csv(combined_path, combined_fields, combined_rows)
    print(f"[format] wrote {combined_path}")

    # Wide pivot
    wide_rows = pivot_wide_paper_table(combined_rows)
    wide_path = out_root / "paper_table_wide.csv"
    wide_fields = [
        "cell",
        "DGP",
        "T",
        "J",
        "Method",
        "Row",
        "Int",
        "beta_p",
        "beta_w",
        "sigma",
        "xi_mean_abs_error",
        "xi_sd_error",
        "Prob_signal",
        "Prob_noise",
        "FailRate",
    ]
    _write_csv(wide_path, wide_fields, wide_rows)
    print(f"[format] wrote {wide_path}")

    # Combine summary.csv if present
    summary_rows, warn2 = combine_summary_long(in_root, grid)
    if warn2:
        for w in warn2:
            print(f"[format][INFO] {w}")

    if summary_rows:
        summary_path = out_root / "summary_long_combined.csv"
        summary_fields = ["cell", "source_path", "dgp", "T", "J", "method", "metric", "value"]
        _write_csv(summary_path, summary_fields, summary_rows)
        print(f"[format] wrote {summary_path}")

    print("[format] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

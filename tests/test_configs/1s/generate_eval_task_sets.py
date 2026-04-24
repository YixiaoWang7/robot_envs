#!/usr/bin/env python3
"""
Generate evaluation config JSON files for CG_L4 task subsets.

We encode tasks as two digits:
- object id 0/1/2/3 => cross/cube/cylinder/milk
- container id 0/1/2/3 => bin/mug/plate/mug_no_handle

This script writes one JSON file per named set into `tests/test_configs/`.
Each output follows the same schema as `template.json` and is meant for
`tests/eval_policy_l4.py` (it uses `train_tasks` to color the 4×4 grid).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence


OBJECTS = ("cross", "cube", "cylinder", "milk")
CONTAINERS = ("bin", "mug", "plate", "mug_no_handle")


SET_DEFS: dict[str, list[str]] = {
    # 4×4 full grid
    "full": [f"{oi}{ci}" for oi in range(4) for ci in range(4)],
    # Provided sets
    "Sfull": ["00", "01", "11", "12", "22", "23", "33", "30"],
    "S": ["00", "01", "11", "12", "22", "23", "33"],
    "L_cor": ["00", "01", "02", "03", "10", "20", "30"],
    "L_mid": ["01", "10", "11", "12", "13", "21", "31"],
    "diag_cor": ["00", "11", "22", "33", "03"],
    "diag_mid": ["00", "11", "22", "33", "02"],
    "diag": ["00", "11", "22", "33"],
}


def _validate_codes(codes: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in codes:
        code = str(raw).strip()
        if not re.fullmatch(r"\d\d", code):
            raise ValueError(f"Invalid code '{raw}': expected two digits like '02'.")
        oi = int(code[0])
        ci = int(code[1])
        if not (0 <= oi < len(OBJECTS)):
            raise ValueError(f"Object index out of range in code '{code}'. Expected 0..3.")
        if not (0 <= ci < len(CONTAINERS)):
            raise ValueError(f"Container index out of range in code '{code}'. Expected 0..3.")
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def codes_to_train_tasks(codes: Sequence[str]) -> list[str]:
    codes = _validate_codes(codes)
    return [f"place the {OBJECTS[int(c[0])]} into the {CONTAINERS[int(c[1])]}" for c in codes]


def load_template(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Template must be a JSON object, got {type(data)} at {path}")
    if "eval_config" not in data or "result_config" not in data:
        raise ValueError(f"Template missing required keys ('eval_config', 'result_config'): {path}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate CG_L4 eval configs (one JSON per named train set).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for generated JSON files (default: ./1s relative to current working directory).",
    )
    ap.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to a template config JSON (defaults to template.json next to this script).",
    )
    ap.add_argument(
        "--sets",
        type=str,
        default=None,
        help=f"Comma-separated set names to write (default: all). Available: {', '.join(sorted(SET_DEFS))}",
    )
    ap.add_argument("--suffix", type=str, default="", help="Optional suffix appended to output filenames.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (Path.cwd() / "1s")
    out_dir.mkdir(parents=True, exist_ok=True)
    template_path = Path(args.template) if args.template else (script_dir / "template.json")
    template = load_template(template_path)

    if args.sets is None:
        set_names = sorted(SET_DEFS.keys())
    else:
        set_names = [s.strip() for s in args.sets.split(",") if s.strip()]
        missing = [s for s in set_names if s not in SET_DEFS]
        if missing:
            raise SystemExit(
                f"Unknown set(s): {missing}. Available: {', '.join(sorted(SET_DEFS.keys()))}"
            )

    wrote = 0
    for set_name in set_names:
        cfg = dict(template)  # shallow copy is enough (we replace top-level train_tasks)
        cfg["train_tasks"] = codes_to_train_tasks(SET_DEFS[set_name])
        cfg.pop("tasks", None)  # avoid ambiguity; eval_policy_l4.py uses full-grid schedule

        name = f"{set_name}{args.suffix}".strip()
        out_path = out_dir / f"{name}.json"
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")
        out_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        wrote += 1

    print(f"Wrote {wrote} config(s) to: {out_dir}")


if __name__ == "__main__":
    main()


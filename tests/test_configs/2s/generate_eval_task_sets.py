#!/usr/bin/env python3
"""
Generate two-stage evaluation config JSON files for CG_L4 task subsets.

Stage-level tasks are encoded as two digits:
- object id 0/1/2/3 => cross/cube/cylinder/milk
- container id 0/1/2/3 => bin/mug/plate/mug_no_handle

Two-stage tasks are formed by combining stage-0 and stage-1 pairs:
  "<obj0>_into_<cont0>__<obj1>_into_<cont1>"

This script writes one JSON file per (stage0_set, stage1_set) pair using a
`template.json` schema and populates:
  - task_slugs
  - stage0_train_tasks / stage0_train_task_slugs
  - stage1_train_tasks / stage1_train_task_slugs
  - constraints

These files are meant for `tests/eval_policy_l4_2s.py`, which requires explicit
`task_slugs` and will validate alignment.
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
    # Provided sets (same as 1s generator)
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


def _code_to_stage_slug(code: str) -> str:
    code = str(code).strip()
    oi = int(code[0])
    ci = int(code[1])
    return f"{OBJECTS[oi]}_into_{CONTAINERS[ci]}"


def codes_to_stage_slugs(codes: Sequence[str]) -> list[str]:
    codes = _validate_codes(codes)
    return [_code_to_stage_slug(c) for c in codes]


def codes_to_stage_tasks(codes: Sequence[str]) -> list[str]:
    codes = _validate_codes(codes)
    return [f"place the {OBJECTS[int(c[0])]} into the {CONTAINERS[int(c[1])]}" for c in codes]


def two_stage_slug(stage0_slug: str, stage1_slug: str) -> str:
    return f"{str(stage0_slug).strip()}__{str(stage1_slug).strip()}"


def generate_two_stage_slugs(
    *,
    stage0_codes: Sequence[str],
    stage1_codes: Sequence[str],
    distinct_objects: bool,
    distinct_containers: bool,
) -> list[str]:
    s0_codes = _validate_codes(stage0_codes)
    s1_codes = _validate_codes(stage1_codes)

    out: list[str] = []
    for c0 in s0_codes:
        o0 = int(c0[0])
        k0 = int(c0[1])
        s0 = _code_to_stage_slug(c0)
        for c1 in s1_codes:
            o1 = int(c1[0])
            k1 = int(c1[1])
            if distinct_objects and o1 == o0:
                continue
            if distinct_containers and k1 == k0:
                continue
            s1 = _code_to_stage_slug(c1)
            out.append(two_stage_slug(s0, s1))

    # stable + unique
    return list(dict.fromkeys(out))


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
        description="Generate CG_L4 two-stage eval configs (one JSON per stage-set pair).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for generated JSON files (default: ./2s relative to current working directory).",
    )
    ap.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to a template config JSON (defaults to template.json next to this script).",
    )
    ap.add_argument(
        "--stage0_sets",
        type=str,
        default=None,
        help=f"Comma-separated stage-0 set names (default: all). Available: {', '.join(sorted(SET_DEFS))}",
    )
    ap.add_argument(
        "--stage1_sets",
        type=str,
        default=None,
        help=f"Comma-separated stage-1 set names (default: all). Available: {', '.join(sorted(SET_DEFS))}",
    )
    ap.add_argument("--suffix", type=str, default="", help="Optional suffix appended to output filenames.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    ap.add_argument("--allow_same_object", action="store_true", help="Allow the same object across stages.")
    ap.add_argument("--allow_same_container", action="store_true", help="Allow the same container across stages.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (Path.cwd() / "2s")
    out_dir.mkdir(parents=True, exist_ok=True)
    template_path = Path(args.template) if args.template else (script_dir / "template.json")
    template = load_template(template_path)

    def _parse_set_list(raw: str | None) -> list[str]:
        if raw is None:
            return sorted(SET_DEFS.keys())
        names = [s.strip() for s in str(raw).split(",") if s.strip()]
        missing = [s for s in names if s not in SET_DEFS]
        if missing:
            raise SystemExit(f"Unknown set(s): {missing}. Available: {', '.join(sorted(SET_DEFS.keys()))}")
        return names

    s0_names = _parse_set_list(args.stage0_sets)
    s1_names = _parse_set_list(args.stage1_sets)

    distinct_objects = not bool(args.allow_same_object)
    distinct_containers = not bool(args.allow_same_container)

    wrote = 0
    for s0 in s0_names:
        for s1 in s1_names:
            cfg = dict(template)  # shallow copy
            cfg["constraints"] = {
                "distinct_objects": bool(distinct_objects),
                "distinct_containers": bool(distinct_containers),
            }

            s0_codes = SET_DEFS[s0]
            s1_codes = SET_DEFS[s1]
            cfg["stage0_train_tasks"] = codes_to_stage_tasks(s0_codes)
            cfg["stage0_train_task_slugs"] = codes_to_stage_slugs(s0_codes)
            cfg["stage1_train_tasks"] = codes_to_stage_tasks(s1_codes)
            cfg["stage1_train_task_slugs"] = codes_to_stage_slugs(s1_codes)
            cfg["task_slugs"] = generate_two_stage_slugs(
                stage0_codes=s0_codes,
                stage1_codes=s1_codes,
                distinct_objects=distinct_objects,
                distinct_containers=distinct_containers,
            )

            name = f"s0_{s0}__s1_{s1}{args.suffix}".strip()
            out_path = out_dir / f"{name}.json"
            if out_path.exists() and not args.overwrite:
                raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")
            out_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            wrote += 1

    print(f"Wrote {wrote} config(s) to: {out_dir}")


if __name__ == "__main__":
    main()


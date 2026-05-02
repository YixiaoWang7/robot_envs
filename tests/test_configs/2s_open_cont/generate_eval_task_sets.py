#!/usr/bin/env python3
"""
Generate open-container two-stage evaluation config JSON files for CG_L4.

Each eval task is keyed by the policy-visible tuple:
  (obj0, cont0, obj1)

The serialized slug still uses the standard two-stage format so existing eval
config validation and dataset naming conventions continue to work. The second
stage container in the slug is a canonical placeholder; the open-container eval
script accepts success in any container except cont0.
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
    "full": [f"{oi}{ci}" for oi in range(4) for ci in range(4)],
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
            raise ValueError(f"Invalid code {raw!r}: expected two digits like '02'.")
        oi = int(code[0])
        ci = int(code[1])
        if not (0 <= oi < len(OBJECTS)):
            raise ValueError(f"Object index out of range in code {code!r}. Expected 0..3.")
        if not (0 <= ci < len(CONTAINERS)):
            raise ValueError(f"Container index out of range in code {code!r}. Expected 0..3.")
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def _code_to_stage_slug(code: str) -> str:
    oi = int(str(code)[0])
    ci = int(str(code)[1])
    return f"{OBJECTS[oi]}_into_{CONTAINERS[ci]}"


def codes_to_stage_slugs(codes: Sequence[str]) -> list[str]:
    return [_code_to_stage_slug(c) for c in _validate_codes(codes)]


def codes_to_stage_tasks(codes: Sequence[str]) -> list[str]:
    codes = _validate_codes(codes)
    return [f"place the {OBJECTS[int(c[0])]} into the {CONTAINERS[int(c[1])]}" for c in codes]


def _canonical_stage1_container(*, obj1: int, cont0: int, stage1_codes: Sequence[str]) -> int:
    for code in _validate_codes(stage1_codes):
        if int(code[0]) == int(obj1) and int(code[1]) != int(cont0):
            return int(code[1])
    for ci in range(len(CONTAINERS)):
        if ci != int(cont0):
            return ci
    raise ValueError("No valid stage-1 container exists.")


def generate_open_container_slugs(
    *,
    stage0_codes: Sequence[str],
    stage1_codes: Sequence[str],
    distinct_objects: bool,
) -> list[str]:
    s0_codes = _validate_codes(stage0_codes)
    s1_codes = _validate_codes(stage1_codes)
    stage1_objects = list(dict.fromkeys(int(c[0]) for c in s1_codes))

    out: list[str] = []
    seen_visible: set[tuple[int, int, int]] = set()
    for c0 in s0_codes:
        obj0 = int(c0[0])
        cont0 = int(c0[1])
        s0 = _code_to_stage_slug(c0)
        for obj1 in stage1_objects:
            if distinct_objects and obj1 == obj0:
                continue
            visible_key = (obj0, cont0, obj1)
            if visible_key in seen_visible:
                continue
            seen_visible.add(visible_key)
            cont1_placeholder = _canonical_stage1_container(
                obj1=obj1,
                cont0=cont0,
                stage1_codes=s1_codes,
            )
            s1 = f"{OBJECTS[obj1]}_into_{CONTAINERS[cont1_placeholder]}"
            out.append(f"{s0}__{s1}")
    return out


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
        description="Generate CG_L4 open-container two-stage eval configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for generated JSON files.",
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
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / "test_configs")
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

    wrote = 0
    for s0 in s0_names:
        for s1 in s1_names:
            cfg = dict(template)
            cfg["constraints"] = {
                "distinct_objects": bool(distinct_objects),
                "stage1_container_rule": "any_except_stage0_container",
            }

            s0_codes = SET_DEFS[s0]
            s1_codes = SET_DEFS[s1]
            cfg["stage0_train_tasks"] = codes_to_stage_tasks(s0_codes)
            cfg["stage0_train_task_slugs"] = codes_to_stage_slugs(s0_codes)
            cfg["stage1_train_tasks"] = codes_to_stage_tasks(s1_codes)
            cfg["stage1_train_task_slugs"] = codes_to_stage_slugs(s1_codes)
            cfg["task_slugs"] = generate_open_container_slugs(
                stage0_codes=s0_codes,
                stage1_codes=s1_codes,
                distinct_objects=distinct_objects,
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

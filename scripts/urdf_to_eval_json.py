#!/usr/bin/env python3
"""
Convert a URDF file into the JSON schema used by this repository's dataset loader.

Output schema keys:
  - point_cloud
  - question
  - normalize
  - answer.links
  - answer.joints
Usage example:
  python scripts/urdf_to_eval_json.py \
    --urdf ./my_robot/mobility.urdf \
    --output ./datasets/my_robot/json_questions/robot01/robot01_0.json \
    --object-name Robot01 \
    --part-map ./my_robot/part_map.json

part_map.json example:
{
  "lid_link": "lid",
  "body_link": "pot_body"
}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_xyz_or_rpy(text: str | None, default: List[float]) -> List[float]:
    if not text:
        return list(default)
    vals = text.strip().split()
    if len(vals) != 3:
        return list(default)
    try:
        return [float(vals[0]), float(vals[1]), float(vals[2])]
    except ValueError:
        return list(default)


def _clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")


def _ordered_child_links(joints: List[Dict]) -> List[str]:
    seen = set()
    ordered = []
    for j in joints:
        c = j["child"]
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _load_part_map(path: str | None) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"part map not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("part map must be a JSON object: {urdf_link_name: part_category}")
    return {str(k): str(v) for k, v in data.items()}


def _resolve_part_name(urdf_link_name: str, part_map: Dict[str, str], allowed_parts: set[str]) -> str:
    if urdf_link_name in part_map:
        mapped = part_map[urdf_link_name]
        return mapped

    guess = _clean_name(urdf_link_name)
    if guess in allowed_parts:
        return guess

    # Fallback for template generation; user can fix via --part-map.
    return "handle"


def convert_urdf_to_schema(
    urdf_path: Path,
    object_name: str,
    part_map: Dict[str, str],
    base_link_name: str,
    include_fixed_joints: bool,
) -> Tuple[Dict, List[str]]:
    # Import list from repo to stay consistent with training/eval loader.
    try:
        from utils.new_dataset import PART_CATEGORIES  # type: ignore
    except Exception:
        PART_CATEGORIES = []

    allowed_parts = set(PART_CATEGORIES)

    root = ET.parse(urdf_path).getroot()

    joints: List[Dict] = []
    for j in root.findall("joint"):
        jtype = j.attrib.get("type", "fixed")
        if (not include_fixed_joints) and jtype == "fixed":
            continue

        parent_el = j.find("parent")
        child_el = j.find("child")
        if parent_el is None or child_el is None:
            continue

        parent = parent_el.attrib.get("link", "base")
        child = child_el.attrib.get("link", "")
        if not child:
            continue

        origin_el = j.find("origin")
        if origin_el is not None:
            xyz = _parse_xyz_or_rpy(origin_el.attrib.get("xyz"), [0.0, 0.0, 0.0])
            rpy = _parse_xyz_or_rpy(origin_el.attrib.get("rpy"), [0.0, 0.0, 0.0])
        else:
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]

        axis_el = j.find("axis")
        axis = _parse_xyz_or_rpy(axis_el.attrib.get("xyz") if axis_el is not None else None, [0.0, 0.0, 1.0])

        limit_el = j.find("limit")
        if limit_el is not None:
            try:
                lower = float(limit_el.attrib.get("lower", 0.0))
            except ValueError:
                lower = 0.0
            try:
                upper = float(limit_el.attrib.get("upper", 0.0))
            except ValueError:
                upper = 0.0
        else:
            lower, upper = 0.0, 0.0

        joints.append(
            {
                "raw_name": j.attrib.get("name", ""),
                "type": jtype,
                "parent": parent,
                "child": child,
                "origin": {"xyz": xyz, "rpy": rpy},
                "axis": axis,
                "limit": {"lower": lower, "upper": upper},
            }
        )

    child_links = _ordered_child_links(joints)
    if not child_links:
        raise RuntimeError("No usable child links found from <joint> entries in URDF.")

    remap: Dict[str, str] = {name: f"link_{idx}" for idx, name in enumerate(child_links)}

    point_cloud: Dict[str, str] = {}
    answer_links: Dict[str, str] = {}
    warnings: List[str] = []
    for raw_link in child_links:
        mapped_link = remap[raw_link]
        part_name = _resolve_part_name(raw_link, part_map, allowed_parts)
        if allowed_parts and part_name not in allowed_parts:
            warnings.append(
                f"part '{part_name}' for URDF link '{raw_link}' is not in PART_CATEGORIES"
            )
        point_cloud[mapped_link] = part_name
        answer_links[mapped_link] = f"{part_name}[SEG]"

    answer_joints: List[Dict] = []
    jidx = 0
    for j in joints:
        child_raw = j["child"]
        if child_raw not in remap:
            continue

        parent_raw = j["parent"]
        if parent_raw == base_link_name:
            parent_mapped = "base"
        elif parent_raw in remap:
            parent_mapped = remap[parent_raw]
        else:
            # Keep chain links that are not part of remap as base to avoid broken references.
            parent_mapped = "base"

        answer_joints.append(
            {
                "id": f"joint_{jidx}",
                "type": j["type"],
                "parent": parent_mapped,
                "child": remap[child_raw],
                "origin": j["origin"],
                "axis": j["axis"],
                "limit": j["limit"],
            }
        )
        jidx += 1

    out = {
        "point_cloud": point_cloud,
        "question": f"This articulated object {object_name} consists of {len(point_cloud)} parts. "
        "Represent the segmentation and joint parameters of all links using a structured JSON object.",
        "normalize": {
            "centroid": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "answer": {
            "links": answer_links,
            "joints": answer_joints,
        },
    }
    return out, warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert URDF to evaluation JSON schema template")
    p.add_argument("--urdf", required=True, help="Input URDF path")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--object-name", default="ArticulatedObject", help="Name used in question text")
    p.add_argument("--part-map", default="", help="Optional JSON map: {urdf_link_name: part_category}")
    p.add_argument("--base-link", default="base", help="Base link name in URDF that should map to 'base'")
    p.add_argument(
        "--include-fixed-joints",
        action="store_true",
        help="Include fixed joints in answer.joints (default: off)",
    )
    p.add_argument(
        "--strict-part-category",
        action="store_true",
        help="Fail if mapped part names are not in PART_CATEGORIES",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    urdf_path = Path(args.urdf)
    out_path = Path(args.output)
    if not urdf_path.exists():
        print(f"[ERROR] URDF not found: {urdf_path}", file=sys.stderr)
        return 2

    part_map = _load_part_map(args.part_map)
    payload, warnings = convert_urdf_to_schema(
        urdf_path=urdf_path,
        object_name=args.object_name,
        part_map=part_map,
        base_link_name=args.base_link,
        include_fixed_joints=args.include_fixed_joints,
    )

    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")
        if args.strict_part_category:
            print("[ERROR] Invalid part names found. Use --part-map to fix them.", file=sys.stderr)
            return 3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    print(f"[OK] Wrote: {out_path}")
    print(f"[INFO] links={len(payload['answer']['links'])}, joints={len(payload['answer']['joints'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

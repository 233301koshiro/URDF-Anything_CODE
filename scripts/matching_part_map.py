#!/usr/bin/env python3
"""
Clean matching_part_map implementation (safe single-file) - use this to generate part_map.json.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

KEYWORD_RULES = {
    "wheel": [("wheel", 100)],
    "tire": [("wheel", 80)],
    "caster": [("caster", 100)],
    "lid": [("lid", 100)],
    "cap": [("cover_lid", 80), ("lid", 70)],
    "cover": [("cover_lid", 90)],
    "door": [("door", 100)],
    "handle": [("handle", 100)],
    "knob": [("knob", 100)],
    "button": [("button", 100)],
    "switch": [("switch", 100)],
    "lever": [("lever", 100)],
    "screen": [("screen", 100), ("display_base", 70)],
    "camera": [("camera_body", 90), ("screen", 50)],
    "lens": [("lens", 100)],
    "sensor": [("sensor", 50)],
    "motor": [("rotation_body", 70)],
    "arm": [("rotation_bar", 70)],
    "leg": [("leg", 100), ("chair_leg", 80)],
    "head": [("head", 100)],
    "base": [("furniture_body", 60), ("box_body", 50)],
    "body": [("furniture_body", 80), ("pot_body", 70)],
    "joint": [("connector", 50)],
    "attachment": [("connector", 60)],
    "bracket": [("connector", 50)],
    "frame": [("furniture_body", 60)],
}

STOP_WORDS = {"link", "joint", "attachment", "bracket", "part", "element", "object"}


def _tokenize_link_name(name: str) -> List[str]:
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    tokens = re.findall(r"[a-z0-9]+", name)
    return tokens


def _score_candidate(link_tokens: List[str], category: str) -> int:
    cat_tokens = _tokenize_link_name(category)
    score = 0
    for token in link_tokens:
        if token in cat_tokens:
            score += 50
    for token in link_tokens:
        if token in KEYWORD_RULES:
            for cand_cat, cand_score in KEYWORD_RULES[token]:
                if cand_cat == category:
                    score += cand_score
    return score


def _infer_part_map(link_names: List[str], allowed_categories: List[str]) -> Dict[str, Tuple[str, int]]:
    result: Dict[str, Tuple[str, int]] = {}
    for link in link_names:
        if link in ("base", "world"):
            continue
        tokens = _tokenize_link_name(link)
        if not tokens or all(t in STOP_WORDS for t in tokens):
            result[link] = ("handle", 0)
            continue
        scores = {cat: _score_candidate(tokens, cat) for cat in allowed_categories}
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]
        if best_score < 10:
            result[link] = ("handle", best_score)
        else:
            result[link] = (best_cat, best_score)
    return result


def extract_links_from_urdf(urdf_path: Path) -> List[str]:
    root = ET.parse(urdf_path).getroot()
    links: List[str] = []
    for link_el in root.findall("link"):
        name = link_el.attrib.get("name", "")
        if name and name not in ("base", "world"):
            links.append(name)
    return links


def load_part_categories() -> List[str]:
    try:
        from utils.new_dataset import PART_CATEGORIES  # type: ignore
        return list(PART_CATEGORIES)
    except Exception:
        print("[WARN] Could not import PART_CATEGORIES from utils.new_dataset. Using fallback.", file=sys.stderr)
        return ["handle", "wheel", "lid", "door", "button"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--urdf', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    urdf_path = Path(args.urdf)
    out_path = Path(args.output)
    if not urdf_path.exists():
        print(f"[ERROR] URDF not found: {urdf_path}")
        return 2
    categories = load_part_categories()
    links = extract_links_from_urdf(urdf_path)
    inferred = _infer_part_map(links, categories)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({k: v[0] for k, v in inferred.items()}, f, indent=2, ensure_ascii=False)
    for link, (cat, score) in inferred.items():
        confidence = 'HIGH' if score >= 50 else 'MED' if score >= 10 else 'LOW'
        print(f"{link:30} -> {cat:20} score={score:3d} {confidence}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
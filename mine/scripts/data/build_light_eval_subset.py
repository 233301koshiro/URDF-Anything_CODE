#!/usr/bin/env python3
import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

PATTERN = re.compile(r"^(?P<obj>\d+)_(?P<minor>\d+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build lightweight eval dataset by keeping one sample per object id (xxxxx_oo -> one xxxxx)."
    )
    p.add_argument("--src-root", default="./datasets/urdf", help="Source dataset root containing json_questions/ and point_clouds/")
    p.add_argument("--dst-root", default="./datasets/urdf_light", help="Destination lightweight dataset root")
    p.add_argument("--preferred-minor", type=int, default=0, help="Preferred minor index oo to keep (fallback to smallest available)")
    p.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place selected files in destination",
    )
    p.add_argument("--with-ply", action="store_true", help="Also include matching .ply point cloud file if present")
    p.add_argument("--max-objects", type=int, default=0, help="Optional cap on number of object ids (0 = no cap)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite destination root if it exists")
    p.add_argument("--dry-run", action="store_true", help="Print what would be created without writing files")
    return p.parse_args()


def _link_or_copy(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    src_json_root = src_root / "json_questions"
    src_pc_root = src_root / "point_clouds"

    if not src_json_root.exists() or not src_pc_root.exists():
        raise FileNotFoundError(
            f"Source root must contain json_questions/ and point_clouds/: {src_root}"
        )

    if dst_root.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(dst_root)

    groups = defaultdict(list)
    for json_path in sorted(src_json_root.glob("*/*.json")):
        m = PATTERN.match(json_path.name)
        if not m:
            continue
        obj_id = m.group("obj")
        minor = int(m.group("minor"))
        base = json_path.stem
        txt_path = src_pc_root / obj_id / f"{base}.txt"
        ply_path = src_pc_root / obj_id / f"{base}.ply"
        if not txt_path.exists():
            continue
        groups[obj_id].append((minor, json_path, txt_path, ply_path if ply_path.exists() else None))

    selected = []
    for obj_id in sorted(groups.keys()):
        candidates = sorted(groups[obj_id], key=lambda x: x[0])
        picked = next((c for c in candidates if c[0] == args.preferred_minor), candidates[0])
        selected.append((obj_id, picked))

    if args.max_objects > 0:
        selected = selected[: args.max_objects]

    n_json = n_txt = n_ply = 0
    for obj_id, (_minor, json_path, txt_path, ply_path) in selected:
        dst_json = dst_root / "json_questions" / obj_id / json_path.name
        dst_txt = dst_root / "point_clouds" / obj_id / txt_path.name
        _link_or_copy(json_path, dst_json, args.link_mode, args.dry_run)
        _link_or_copy(txt_path, dst_txt, args.link_mode, args.dry_run)
        n_json += 1
        n_txt += 1
        if args.with_ply and ply_path is not None:
            dst_ply = dst_root / "point_clouds" / obj_id / ply_path.name
            _link_or_copy(ply_path, dst_ply, args.link_mode, args.dry_run)
            n_ply += 1

    print("=== Lightweight subset summary ===")
    print(f"source_root: {src_root}")
    print(f"dest_root:   {dst_root}")
    print(f"objects_selected: {len(selected)}")
    print(f"json_files: {n_json}")
    print(f"txt_files:  {n_txt}")
    print(f"ply_files:  {n_ply}")
    print(f"mode: {args.link_mode}")
    print(f"preferred_minor: {args.preferred_minor}")
    print(f"dry_run: {args.dry_run}")


if __name__ == "__main__":
    main()

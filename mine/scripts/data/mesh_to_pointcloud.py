#!/usr/bin/env python3
"""
Generate point cloud .txt files from URDF geometry (mesh or primitive shapes).

Usage:
  python scripts/mesh_to_pointcloud.py \
    --urdf ./mine/rrbot_description/urdf/rrbot.urdf \
    --output-dir ./datasets/rrbot_test/point_clouds \
    --part-map ./mine/rrbot_part_map.json \
    --object-id rrbot_test

Supports: mesh (DAE, OBJ, STL), box, cylinder, sphere geometries.

Format of output .txt: obj_id part_name x y z r g b inst_code
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import trimesh
except ImportError:
    print("[ERROR] Please install trimesh and numpy: pip install trimesh numpy")
    sys.exit(1)


def resolve_mesh_path(mesh_filename: str, urdf_dir: Path) -> Optional[Path]:
    """Resolve mesh path from filename."""
    if mesh_filename.startswith("file://"):
        mesh_path = Path(mesh_filename[7:])
    elif mesh_filename.startswith("package://"):
        parts = mesh_filename[10:].split("/", 1)
        mesh_path = Path("/opt/ros") / parts[0] / parts[1] if len(parts) > 1 else None
    else:
        mesh_path = urdf_dir / mesh_filename
    
    if mesh_path and mesh_path.exists():
        return mesh_path
    return None


def sample_mesh_points(mesh_path: Path, num_points: int = 2048) -> Optional[np.ndarray]:
    """Sample points uniformly from mesh surface."""
    try:
        mesh = trimesh.load(str(mesh_path))
        if mesh.is_empty:
            return None
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except Exception as e:
        print(f"[WARN] Could not sample {mesh_path}: {e}", file=sys.stderr)
        return None


def sample_box_points(size: List[float], num_points: int) -> np.ndarray:
    """Sample points uniformly from box surface."""
    x, y, z = size[0] / 2, size[1] / 2, size[2] / 2
    vertices = np.array([
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2]
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def sample_cylinder_points(radius: float, length: float, num_points: int) -> np.ndarray:
    """Sample points from cylinder surface."""
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=32)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def sample_sphere_points(radius: float, num_points: int) -> np.ndarray:
    """Sample points from sphere surface."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def extract_geometry_from_urdf(urdf_path: Path) -> Dict[str, Tuple[str, Optional[Path], Optional[Dict]]]:
    """Extract link name -> (geometry_type, mesh_path, params) from URDF."""
    root = ET.parse(urdf_path).getroot()
    urdf_dir = urdf_path.parent
    result = {}
    
    for link_el in root.findall("link"):
        link_name = link_el.attrib.get("name", "")
        if not link_name:
            continue
        
        geom_el = link_el.find(".//geometry")
        if geom_el is None:
            result[link_name] = ("none", None, None)
            continue
        
        mesh_el = geom_el.find("mesh")
        if mesh_el is not None:
            filename = mesh_el.attrib.get("filename", "")
            if filename:
                resolved = resolve_mesh_path(filename, urdf_dir)
                result[link_name] = ("mesh", resolved, None)
                continue
        
        box_el = geom_el.find("box")
        if box_el is not None:
            size_str = box_el.attrib.get("size", "1 1 1")
            try:
                size = [float(x) for x in size_str.split()]
                result[link_name] = ("box", None, {"size": size})
                continue
            except:
                pass
        
        cyl_el = geom_el.find("cylinder")
        if cyl_el is not None:
            try:
                radius = float(cyl_el.attrib.get("radius", 1.0))
                length = float(cyl_el.attrib.get("length", 1.0))
                result[link_name] = ("cylinder", None, {"radius": radius, "length": length})
                continue
            except:
                pass
        
        sphere_el = geom_el.find("sphere")
        if sphere_el is not None:
            try:
                radius = float(sphere_el.attrib.get("radius", 1.0))
                result[link_name] = ("sphere", None, {"radius": radius})
                continue
            except:
                pass
        
        result[link_name] = ("none", None, None)
    
    return result


def sample_geometry_points(geom_type: str, mesh_path: Optional[Path], params: Optional[Dict], num_points: int = 2048) -> Optional[np.ndarray]:
    """Sample points from various geometry types."""
    if geom_type == "mesh" and mesh_path:
        return sample_mesh_points(mesh_path, num_points)
    elif geom_type == "box" and params:
        return sample_box_points(params["size"], num_points)
    elif geom_type == "cylinder" and params:
        return sample_cylinder_points(params["radius"], params["length"], num_points)
    elif geom_type == "sphere" and params:
        return sample_sphere_points(params["radius"], num_points)
    return None


def load_part_map(part_map_path: Path) -> Dict[str, str]:
    """Load part_map.json."""
    try:
        with part_map_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load part_map: {e}", file=sys.stderr)
        return {}


def generate_pointcloud_txt(link_name: str, part_name: str, points: np.ndarray, object_id: str, instance_id: int = 0) -> List[str]:
    """Generate .txt lines for point cloud."""
    lines = []
    for i, (x, y, z) in enumerate(points):
        inst_code = 1 if i % 10 == instance_id % 10 else 0
        line = f"{object_id} {part_name} {x:.6f} {y:.6f} {z:.6f} 1.0 1.0 1.0 {inst_code}"
        lines.append(line)
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description="Generate point cloud .txt from URDF geometry")
    p.add_argument("--urdf", required=True, help="Input URDF path")
    p.add_argument("--output-dir", required=True, help="Output directory for point_clouds")
    p.add_argument("--part-map", required=True, help="Path to part_map.json")
    p.add_argument("--object-id", default="test", help="Object ID to use in output")
    p.add_argument("--num-points", type=int, default=2048, help="Number of points to sample per geometry")
    args = p.parse_args()
    
    urdf_path = Path(args.urdf)
    out_dir = Path(args.output_dir)
    part_map_path = Path(args.part_map)
    
    if not urdf_path.exists():
        print(f"[ERROR] URDF not found: {urdf_path}")
        return 2
    if not part_map_path.exists():
        print(f"[ERROR] part_map not found: {part_map_path}")
        return 2
    
    link_geoms = extract_geometry_from_urdf(urdf_path)
    part_map = load_part_map(part_map_path)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_lines = []
    found_geoms = 0
    
    for link_name, (geom_type, mesh_path, params) in sorted(link_geoms.items()):
        part_name = part_map.get(link_name, "handle")
        
        if geom_type == "none":
            print(f"[INFO] {link_name}: no geometry found, skipping")
            continue
        
        print(f"[INFO] Sampling {link_name} ({geom_type}) -> {part_name}")
        points = sample_geometry_points(geom_type, mesh_path, params, args.num_points)
        
        if points is None:
            print(f"[WARN] Failed to sample {link_name}, skipping")
            continue
        
        lines = generate_pointcloud_txt(link_name, part_name, points, args.object_id, instance_id=found_geoms)
        all_lines.extend(lines)
        found_geoms += 1
    
    if not all_lines:
        print("[ERROR] No point clouds generated")
        return 1
    
    out_file = out_dir / f"{args.object_id}_0.txt"
    with out_file.open("w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")
    
    print(f"\n[OK] Wrote {len(all_lines)} points to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Generate point clouds (.txt) from URDF geometry.
Creates synthetic point clouds with part names and instance one-hot encoding.
"""

import argparse
import json
import random
from math import cos, sin
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import xml.etree.ElementTree as ET
import colorsys


def _parse_xyz(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return np.array(default, dtype=float)
    vals = text.strip().split()
    if len(vals) != 3:
        return np.array(default, dtype=float)
    try:
        return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=float)
    except ValueError:
        return np.array(default, dtype=float)


def _rpy_to_matrix(rpy):
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def _make_transform(xyz, rpy):
    T = np.eye(4, dtype=float)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return points @ T[:3, :3].T + T[:3, 3]


def extract_geometry_from_urdf(urdf_path: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Extract geometry information from URDF file.
    Returns:
      geometries: {link_name: {geometry_type, params, origin_xyz, origin_rpy}}
      joints: {child_link: {parent, origin_xyz, origin_rpy, type}}
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    geometries = {}
    joints = {}
    
    for link in root.findall('link'):
        link_name = link.get('name')
        
        # Try to find visual geometry (preferred over collision)
        visual = link.find('.//visual')
        if visual is None:
            visual = link.find('.//collision')
        
        if visual is None:
            continue
        
        geom = visual.find('geometry')
        if geom is None:
            continue

        origin_el = visual.find('origin')
        origin_xyz = _parse_xyz(origin_el.get('xyz') if origin_el is not None else None)
        origin_rpy = _parse_xyz(origin_el.get('rpy') if origin_el is not None else None)
        
        # Extract geometry type and parameters
        if geom.find('box') is not None:
            box = geom.find('box')
            size = [float(x) for x in box.get('size').split()]
            geometries[link_name] = {
                'geometry_type': 'box',
                'params': {'size': size},
                'origin_xyz': origin_xyz,
                'origin_rpy': origin_rpy,
            }
        elif geom.find('cylinder') is not None:
            cyl = geom.find('cylinder')
            radius = float(cyl.get('radius'))
            length = float(cyl.get('length'))
            geometries[link_name] = {
                'geometry_type': 'cylinder',
                'params': {'radius': radius, 'length': length},
                'origin_xyz': origin_xyz,
                'origin_rpy': origin_rpy,
            }
        elif geom.find('sphere') is not None:
            sphere = geom.find('sphere')
            radius = float(sphere.get('radius'))
            geometries[link_name] = {
                'geometry_type': 'sphere',
                'params': {'radius': radius},
                'origin_xyz': origin_xyz,
                'origin_rpy': origin_rpy,
            }
        else:
            # Default to small box for mesh/unknown
            geometries[link_name] = {
                'geometry_type': 'box',
                'params': {'size': [0.05, 0.05, 0.05]},
                'origin_xyz': origin_xyz,
                'origin_rpy': origin_rpy,
            }

    for joint in root.findall('joint'):
        parent_el = joint.find('parent')
        child_el = joint.find('child')
        if parent_el is None or child_el is None:
            continue
        child_name = child_el.get('link')
        if not child_name:
            continue
        origin_el = joint.find('origin')
        origin_xyz = _parse_xyz(origin_el.get('xyz') if origin_el is not None else None)
        origin_rpy = _parse_xyz(origin_el.get('rpy') if origin_el is not None else None)
        joints[child_name] = {
            'parent': parent_el.get('link', 'world'),
            'origin_xyz': origin_xyz,
            'origin_rpy': origin_rpy,
            'type': joint.get('type', 'fixed'),
        }
    
    return geometries, joints


def generate_box_points(size: List[float], num_points: int, origin: List[float]) -> np.ndarray:
    """Generate random points within a box."""
    x, y, z = size
    origin = np.array(origin)
    
    # Generate points in box centered at origin
    points = np.random.uniform(
        low=[-x/2, -y/2, -z/2],
        high=[x/2, y/2, z/2],
        size=(num_points, 3)
    ) + origin
    
    return points


def generate_cylinder_points(radius: float, length: float, num_points: int, origin: List[float]) -> np.ndarray:
    """Generate random points within a cylinder (Z-axis aligned)."""
    origin = np.array(origin)
    
    # Generate points in cylinder
    r = np.sqrt(np.random.uniform(0, 1, num_points)) * radius
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-length/2, length/2, num_points)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack([x, y, z]) + origin
    return points


def generate_sphere_points(radius: float, num_points: int, origin: List[float]) -> np.ndarray:
    """Generate random points within a sphere."""
    origin = np.array(origin)
    
    # Generate points in sphere using rejection sampling
    points = []
    while len(points) < num_points:
        candidate = np.random.uniform(
            low=[-radius, -radius, -radius],
            high=[radius, radius, radius]
        )
        if np.linalg.norm(candidate) <= radius:
            points.append(candidate)
    
    points = np.array(points[:num_points]) + origin
    return points


def generate_color(seed: int) -> Tuple[int, int, int]:
    """Generate deterministic RGB color from link name hash."""
    random.seed(seed)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def generate_distinct_color(index: int, total: int) -> Tuple[int, int, int]:
    """Generate a distinct RGB color by evenly spacing hues in HSV space.

    - `index` should be in [0, total-1].
    - returns (R,G,B) in 0-255 ints.
    """
    if total <= 0:
        return (200, 200, 200)
    # Evenly spaced hue in [0,1)
    h = (float(index) / float(total)) % 1.0
    s = 0.80
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def generate_one_hot_codes(instance_id: int, num_links: int) -> np.ndarray:
    """Generate one-hot encoding for instance ID."""
    codes = np.zeros(num_links, dtype=int)
    if 0 <= instance_id < num_links:
        codes[instance_id] = 1
    return codes


def _stable_color_seed(name: str) -> int:
    seed = 0
    for i, ch in enumerate(name):
        seed = (seed + (i + 1) * ord(ch)) % (2**31 - 1)
    return seed


def _build_link_world_transform(link_name: str, joints: Dict[str, dict], cache: Dict[str, np.ndarray]) -> np.ndarray:
    if link_name in cache:
        return cache[link_name]

    joint = joints.get(link_name)
    if joint is None:
        cache[link_name] = np.eye(4, dtype=float)
        return cache[link_name]

    parent = joint['parent']
    if parent in (None, '', 'world'):
        parent_T = np.eye(4, dtype=float)
    else:
        parent_T = _build_link_world_transform(parent, joints, cache)

    joint_T = _make_transform(joint['origin_xyz'], joint['origin_rpy'])
    cache[link_name] = parent_T @ joint_T
    return cache[link_name]


def create_pointcloud_txt(
    urdf_path: str,
    part_map_path: str,
    output_dir: str,
    output_filename: str = "points.txt",
    points_per_link: int = 1000,
    seed: int = 42
) -> None:
    """
    Generate point cloud .txt file from URDF.
    Format: obj_id part_name x y z r g b inst_code1 inst_code2 ...
    
    Args:
        urdf_path: Path to URDF file
        part_map_path: Path to part_map.json
        output_dir: Output directory
        output_filename: Output filename (default: "points.txt")
        points_per_link: Points per link
        seed: Random seed
    """
    np.random.seed(seed)
    random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load geometries and joint chain from URDF
    geometries, joints = extract_geometry_from_urdf(urdf_path)
    
    # Load part map
    part_map = {}
    if Path(part_map_path).exists():
        with open(part_map_path) as f:
            part_map = json.load(f)
    
    # Get sorted link names (excluding base/world)
    link_names = sorted([ln for ln in geometries.keys() if ln not in ('base', 'world')])
    num_links = len(link_names)
    
    if num_links == 0:
        print("[ERROR] No links found in URDF")
        return
    
    print(f"[INFO] Found {num_links} links: {link_names}")
    
    # Generate point cloud
    all_points = []
    world_cache = {}
    
    for link_idx, link_name in enumerate(link_names):
        geom_info = geometries[link_name]
        geom_type = geom_info['geometry_type']
        params = geom_info['params']
        geom_origin_xyz = geom_info['origin_xyz']
        geom_origin_rpy = geom_info['origin_rpy']
        
        # Get part category from part_map (default: link name)
        part_category = part_map.get(link_name, link_name)
        
        # Generate points for this link in the local geometry frame
        if geom_type == 'box':
            points = generate_box_points(params['size'], points_per_link, [0, 0, 0])
        elif geom_type == 'cylinder':
            points = generate_cylinder_points(params['radius'], params['length'], points_per_link, [0, 0, 0])
        elif geom_type == 'sphere':
            points = generate_sphere_points(params['radius'], points_per_link, [0, 0, 0])
        else:
            points = generate_box_points([0.1, 0.1, 0.1], points_per_link, [0, 0, 0])

        # Apply visual/collision origin and the URDF joint chain
        geom_T = _make_transform(geom_origin_xyz, geom_origin_rpy)
        link_world_T = _build_link_world_transform(link_name, joints, world_cache)
        points = _apply_transform(points, link_world_T @ geom_T)
        
        # Generate color: use evenly spaced hues for visually distinct parts
        color = generate_distinct_color(link_idx, num_links)
        
        # Generate one-hot codes
        one_hot = generate_one_hot_codes(link_idx, num_links)
        
        # Add to point cloud: [obj_id, part_name, x, y, z, r, g, b, inst_code1, ...]
        for point in points:
            x, y, z = point
            row = ['0', part_category, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"] + \
                  [str(c) for c in color] + \
                  [str(code) for code in one_hot]
            all_points.append(row)
    
    # Write to txt file
    output_file = output_path / output_filename
    with open(output_file, 'w') as f:
        for row in all_points:
            f.write(' '.join(row) + '\n')
    
    print(f"[OK] Generated point cloud: {output_file}")
    print(f"[INFO] Total points: {len(all_points)}")
    
    # Create metadata
    metadata = {
        "num_links": num_links,
        "link_names": link_names,
        "part_map": part_map,
        "num_points": len(all_points),
        "points_per_link": points_per_link
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Wrote metadata: {metadata_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate point clouds from URDF')
    parser.add_argument('--urdf', required=True, help='Path to URDF file')
    parser.add_argument('--part-map', required=True, help='Path to part_map.json')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--output-filename', type=str, default='points.txt', help='Output filename (default: points.txt)')
    parser.add_argument('--points-per-link', type=int, default=1000, help='Number of points per link')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_pointcloud_txt(
        urdf_path=args.urdf,
        part_map_path=args.part_map,
        output_dir=args.output,
        output_filename=args.output_filename,
        points_per_link=args.points_per_link,
        seed=args.seed
    )

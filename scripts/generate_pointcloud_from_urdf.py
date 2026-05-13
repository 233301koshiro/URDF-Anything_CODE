#!/usr/bin/env python3
"""
Generate point clouds (.txt) from URDF geometry.
Creates synthetic point clouds with part names and instance one-hot encoding.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import xml.etree.ElementTree as ET


def extract_geometry_from_urdf(urdf_path: str) -> Dict[str, Tuple[str, dict]]:
    """
    Extract geometry information from URDF file.
    Returns: {link_name: (geometry_type, params)}
      - geometry_type: 'box', 'cylinder', 'sphere'
      - params: {'size': [x,y,z]} for box, {'radius': r, 'length': l} for cylinder
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    geometries = {}
    
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
        
        # Extract geometry type and parameters
        if geom.find('box') is not None:
            box = geom.find('box')
            size = [float(x) for x in box.get('size').split()]
            geometries[link_name] = ('box', {'size': size})
        elif geom.find('cylinder') is not None:
            cyl = geom.find('cylinder')
            radius = float(cyl.get('radius'))
            length = float(cyl.get('length'))
            geometries[link_name] = ('cylinder', {'radius': radius, 'length': length})
        elif geom.find('sphere') is not None:
            sphere = geom.find('sphere')
            radius = float(sphere.get('radius'))
            geometries[link_name] = ('sphere', {'radius': radius})
        else:
            # Default to small box for mesh/unknown
            geometries[link_name] = ('box', {'size': [0.05, 0.05, 0.05]})
    
    return geometries


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


def generate_one_hot_codes(instance_id: int, num_links: int) -> np.ndarray:
    """Generate one-hot encoding for instance ID."""
    codes = np.zeros(num_links, dtype=int)
    if 0 <= instance_id < num_links:
        codes[instance_id] = 1
    return codes


def create_pointcloud_txt(
    urdf_path: str,
    part_map_path: str,
    output_dir: str,
    points_per_link: int = 1000,
    seed: int = 42
) -> None:
    """
    Generate point cloud .txt file from URDF.
    Format: obj_id part_name x y z r g b inst_code1 inst_code2 ...
    """
    np.random.seed(seed)
    random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load geometries from URDF
    geometries = extract_geometry_from_urdf(urdf_path)
    
    # Load part map
    part_map = {}
    if Path(part_map_path).exists():
        with open(part_map_path) as f:
            part_map = json.load(f)
    
    # Get sorted link names (excluding base)
    link_names = sorted([ln for ln in geometries.keys() if ln != 'base'])
    num_links = len(link_names)
    
    if num_links == 0:
        print("[ERROR] No links found in URDF")
        return
    
    print(f"[INFO] Found {num_links} links: {link_names}")
    
    # Generate point cloud
    all_points = []
    
    for link_idx, link_name in enumerate(link_names):
        geom_type, params = geometries[link_name]
        
        # Get part category from part_map (default: link name)
        part_category = part_map.get(link_name, link_name)
        
        # Generate points for this link
        origin = [0, 0, 0]  # Simplified: all at origin (real implementation would use link origins)
        
        if geom_type == 'box':
            points = generate_box_points(params['size'], points_per_link, origin)
        elif geom_type == 'cylinder':
            points = generate_cylinder_points(params['radius'], params['length'], points_per_link, origin)
        elif geom_type == 'sphere':
            points = generate_sphere_points(params['radius'], points_per_link, origin)
        else:
            points = generate_box_points([0.1, 0.1, 0.1], points_per_link, origin)
        
        # Generate color (deterministic from link name)
        color = generate_color(hash(link_name) % (2**31))
        
        # Generate one-hot codes
        one_hot = generate_one_hot_codes(link_idx, num_links)
        
        # Add to point cloud: [obj_id, part_name, x, y, z, r, g, b, inst_code1, ...]
        for point in points:
            x, y, z = point
            row = [0, part_category, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"] + \
                  [str(c) for c in color] + \
                  [str(code) for code in one_hot]
            all_points.append(row)
    
    # Write to txt file
    output_file = output_path / "points.txt"
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
    parser.add_argument('--points-per-link', type=int, default=1000, help='Number of points per link')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_pointcloud_txt(
        urdf_path=args.urdf,
        part_map_path=args.part_map,
        output_dir=args.output,
        points_per_link=args.points_per_link,
        seed=args.seed
    )

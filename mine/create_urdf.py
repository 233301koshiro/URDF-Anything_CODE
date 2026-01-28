import os
import json
import argparse
import numpy as np
import xml.etree.ElementTree as ET

def try_import_open3d():
    try:
        import open3d as o3d
        return o3d
    except Exception:
        return None

def load_structure(structure_json_path: str):
    with open(structure_json_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    assert "joints" in s and "links" in s, "structure json must have keys: joints, links"
    return s

def load_part_pointclouds(parts_dir: str, link_names):
    pcs = {}
    for ln in link_names:
        p = os.path.join(parts_dir, f"{ln}.npy")
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        if arr.ndim != 2 or arr.shape[1] not in (3, 6):
            raise ValueError(f"{p}: expected Nx3 or Nx6 npy, got {arr.shape}")
        xyz = arr[:, :3].astype(np.float64)
        rgb = None
        if arr.shape[1] == 6:
            rgb = arr[:, 3:6].astype(np.float64)
        pcs[ln] = (xyz, rgb)
    return pcs

def pointcloud_to_mesh_obj_open3d(o3d, xyz: np.ndarray, out_obj_path: str):
    # Ball Pivoting: requires normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(50)

    # heuristic radii based on point spacing
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    if len(dists) == 0:
        raise ValueError("point cloud is empty")
    avg = float(np.mean(dists))
    radii = [avg * 1.5, avg * 2.5, avg * 4.0]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
    o3d.io.write_triangle_mesh(out_obj_path, mesh)
    return out_obj_path

def bbox_size_from_xyz(xyz: np.ndarray):
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    size = (mx - mn)
    # avoid zero-size boxes
    size = np.maximum(size, 1e-4)
    return size.tolist()
def build_urdf(structure, mesh_map, geometry_fallback_box_map,
               out_urdf_path: str, robot_name="urdf_anything_like",
               default_effort=1.0, default_velocity=1.0):
    robot = ET.Element("robot", {"name": robot_name})

    # Links
    for link_name in structure["links"].keys():
        link_el = ET.SubElement(robot, "link", {"name": link_name})

        # visual
        vis = ET.SubElement(link_el, "visual")
        geom = ET.SubElement(vis, "geometry")

        if link_name in mesh_map:
            ET.SubElement(geom, "mesh", {"filename": mesh_map[link_name]})
        else:
            # fallback to box primitive
            size = geometry_fallback_box_map.get(link_name, [0.1, 0.1, 0.1])
            ET.SubElement(geom, "box", {"size": f"{size[0]} {size[1]} {size[2]}"})

        # collision (mirror visual)
        col = ET.SubElement(link_el, "collision")
        cgeom = ET.SubElement(col, "geometry")
        if link_name in mesh_map:
            ET.SubElement(cgeom, "mesh", {"filename": mesh_map[link_name]})
        else:
            size = geometry_fallback_box_map.get(link_name, [0.1, 0.1, 0.1])
            ET.SubElement(cgeom, "box", {"size": f"{size[0]} {size[1]} {size[2]}"})

    # Joints
    for j in structure["joints"]:
        jname = j["id"]
        jtype = j["type"]
        parent = j["parent"]
        child = j["child"]

        joint_el = ET.SubElement(robot, "joint", {"name": jname, "type": jtype})
        ET.SubElement(joint_el, "parent", {"link": parent})
        ET.SubElement(joint_el, "child", {"link": child})

        xyz = j["origin"]["xyz"]
        rpy = j["origin"]["rpy"]
        ET.SubElement(
            joint_el, "origin",
            {"xyz": f"{xyz[0]} {xyz[1]} {xyz[2]}", "rpy": f"{rpy[0]} {rpy[1]} {rpy[2]}"}
        )

        axis = j.get("axis", [0, 0, 0])
        ET.SubElement(joint_el, "axis", {"xyz": f"{axis[0]} {axis[1]} {axis[2]}"})

        # limit: Choreonoid等のため effort/velocity を必ず出す（revolute/prismatic/continuous）
        if jtype in ("revolute", "prismatic", "continuous"):
            lim = j.get("limit", {}) or {}

            attrs = {}

            # revolute/prismatic は lower/upper があると嬉しい（無いなら省略可）
            if jtype in ("revolute", "prismatic"):
                if "lower" in lim:
                    attrs["lower"] = str(lim["lower"])
                if "upper" in lim:
                    attrs["upper"] = str(lim["upper"])

            # effort/velocity は必須寄りなのでデフォルト補完
            attrs["effort"] = str(lim.get("effort", default_effort))
            attrs["velocity"] = str(lim.get("velocity", default_velocity))

            ET.SubElement(joint_el, "limit", attrs)

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ", level=0)
    with open(out_urdf_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default="./output/ur3/merge_fixed_joint_ur3_gripper_structure.json", help="structure json with joints/links")
    ap.add_argument("--parts_dir", default="./output/ur3/merge_fixed_joint_ur3_gripper_labeled.npy", help="directory containing <link>.npy")
    ap.add_argument("--out_urdf", default="./output/ur3/ua.urdf", help="output urdf path")
    ap.add_argument("--mesh_dir", default="ua_meshes", help="directory to export meshes")
    ap.add_argument("--robot_name", default="urdf_anything_like")
    args = ap.parse_args()

    structure = load_structure(args.structure)
    link_names = list(structure["links"].keys())

    pcs = load_part_pointclouds(args.parts_dir, link_names)

    o3d = try_import_open3d()
    mesh_map = {}
    box_map = {}

    for ln in link_names:
        if ln not in pcs:
            # no point cloud -> fallback box
            box_map[ln] = [0.1, 0.1, 0.1]
            continue

        xyz, _ = pcs[ln]
        box_map[ln] = bbox_size_from_xyz(xyz)

        if o3d is None:
            # no open3d -> box fallback
            continue

        out_obj = os.path.join(args.mesh_dir, f"{ln}.obj")
        try:
            pointcloud_to_mesh_obj_open3d(o3d, xyz, out_obj)
            # URDF mesh path: keep relative if possible
            mesh_map[ln] = out_obj
        except Exception:
            # if meshing fails, fallback box
            pass

    build_urdf(structure, mesh_map, box_map, args.out_urdf, robot_name=args.robot_name)
    print(f"Wrote URDF: {args.out_urdf}")
    if o3d is None:
        print("open3d not found -> used box fallback (no mesh export).")
    else:
        print(f"Exported meshes to: {args.mesh_dir} (when meshing succeeded)")

if __name__ == "__main__":
    main()

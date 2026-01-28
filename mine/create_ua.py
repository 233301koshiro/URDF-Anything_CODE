import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import yourdfpy
import trimesh
#点群がパーツごとにバラバラになる問題があったが、secene.dumpという、
#trimeshが提供する「SceneGraph を評価して、各 geometry に正しい変換を適用した “ワールド座標系の Trimesh（の集合）” を返す」関数
#を使うことで解決できた。

# =============================================================================
# Part 1: 構造パラメータ(JSON)抽出（そのまま残す）
# =============================================================================
def parse_urdf_to_structure_json(urdf_path, output_json_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints_data = []
    links_map = {}

    for link in root.findall('link'):
        link_name = link.get('name')
        links_map[link_name] = "generic_part [SEG]"

    for joint in root.findall('joint'):
        joint_dict = {
            "id": joint.get('name'), "type": joint.get('type'),
            "parent": joint.find('parent').get('link'),
            "child": joint.find('child').get('link'),
            "origin": {"xyz": [0,0,0], "rpy": [0,0,0]}, "axis": [0,0,0]
        }

        origin = joint.find('origin')
        if origin is not None:
            if origin.get('xyz'):
                joint_dict["origin"]["xyz"] = [float(x) for x in origin.get('xyz').split()]
            if origin.get('rpy'):
                joint_dict["origin"]["rpy"] = [float(x) for x in origin.get('rpy').split()]

        axis = joint.find('axis')
        if axis is not None and axis.get('xyz'):
            joint_dict["axis"] = [float(x) for x in axis.get('xyz').split()]

        limit = joint.find('limit')
        if limit is not None:
            joint_dict["limit"] = {}
            if limit.get('lower'): joint_dict["limit"]["lower"] = float(limit.get('lower'))
            if limit.get('upper'): joint_dict["limit"]["upper"] = float(limit.get('upper'))

        joints_data.append(joint_dict)

    with open(output_json_path, "w") as f:
        json.dump({"joints": joints_data, "links": links_map}, f, indent=4)


# =============================================================================
# dump出力の型ゆれを吸収
# =============================================================================
def _as_named_meshes(dumped):
    """
    returns: list of (name_or_none, trimesh.Trimesh)
    """
    if dumped is None:
        return []
    if isinstance(dumped, trimesh.Trimesh):
        return [(None, dumped)]
    if isinstance(dumped, dict):
        out = []
        for k, v in dumped.items():
            out.extend([(k, m) for _, m in _as_named_meshes(v)])
        return out
    if isinstance(dumped, (list, tuple)):
        out = []
        for v in dumped:
            out.extend(_as_named_meshes(v))
        return out
    return []

def _finite_mesh(m: trimesh.Trimesh) -> bool:
    if m is None or not hasattr(m, "vertices") or len(m.vertices) == 0:
        return False
    return np.isfinite(np.asarray(m.vertices)).all()

# =============================================================================
# Part 2: ラベル付き点群（scene.dumpベース：バラけない）
# =============================================================================
def generate_labeled_pointcloud_from_scene_dump(
    urdf_path,
    output_dir,
    samples_per_mesh=2048,
):
    robot = yourdfpy.URDF.load(urdf_path, load_meshes=True, load_collision_meshes=False)
    robot.update_cfg(configuration={j: 0.0 for j in robot.joint_map})

    scene = robot.scene
    link_names = list(robot.link_map.keys())
    link_set = set(link_names)
    link_name_to_id = {n: i for i, n in enumerate(link_names)}

    # node -> owner_link（親たどり）
    def guess_owner_link(node):
        cur = node
        for _ in range(80):
            if cur in link_set:
                return cur
            p = scene.graph.transforms.parents.get(cur)
            if p is None:
                break
            cur = p
        return None

    # node <-> geom_name 対応
    node_to_geom = {}
    geom_to_nodes = {}
    for node in list(scene.graph.nodes):
        try:
            geom = scene.graph[node][1]
        except Exception:
            geom = None
        if geom is None:
            continue
        node_to_geom[node] = geom
        geom_to_nodes.setdefault(geom, []).append(node)

    # 1) dumpで「ワールド化済み」メッシュ群を得る（これが正）
    try:
        dumped = scene.dump(concatenate=False)
    except TypeError:
        dumped = scene.dump()

    named_meshes = _as_named_meshes(dumped)
    named_meshes = [(k, m) for (k, m) in named_meshes if _finite_mesh(m)]

    if len(named_meshes) == 0:
        raise RuntimeError("scene.dump() returned no valid meshes. (unexpected)")

    # 2) 各メッシュにラベルを割り当ててサンプル
    all_points, all_colors, all_labels = [], [], []

    # dumpが名前付きで返る場合はそれを使って所属ノードを推定
    #  - key が node名なら node->link
    #  - key が geom名なら geom->node->link
    #  - key が無いなら nodes_geometry と順番合わせ（最後の手段）
    nodes_geom = list(getattr(scene.graph, "nodes_geometry", []))

    for idx, (key, mesh_world) in enumerate(named_meshes):
        owner_link = None

        if isinstance(key, str):
            if key in node_to_geom:
                owner_link = guess_owner_link(key)
            elif key in geom_to_nodes:
                owner_link = guess_owner_link(geom_to_nodes[key][0])

        if owner_link is None and idx < len(nodes_geom):
            # 最後の手段：順番が一致している前提で合わせる（環境依存）
            owner_link = guess_owner_link(nodes_geom[idx])

        if owner_link is None:
            # ラベル取れないが点群は作れる
            continue

        label_id = link_name_to_id[owner_link]

        pts, _ = trimesh.sample.sample_surface(mesh_world, samples_per_mesh)
        all_points.append(pts)
        all_colors.append(np.ones((len(pts), 3)))
        all_labels.append(np.full(len(pts), label_id))

    if not all_points:
        raise RuntimeError("Could not assign any labels from dump meshes. (key mapping failed)")

    X = np.vstack(all_points)
    C = np.vstack(all_colors)
    L = np.concatenate(all_labels).reshape(-1, 1)

    # 正規化
    centroid = np.mean(X, axis=0)
    X -= centroid
    max_dist = np.max(np.linalg.norm(X, axis=1))
    if max_dist > 0:
        X /= max_dist

    final = np.hstack((X, C, L)).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(urdf_path))[0]
    np.save(os.path.join(output_dir, f"{base}_labeled.npy"), final)
    with open(os.path.join(output_dir, f"{base}_label_map.json"), "w") as f:
        json.dump(link_name_to_id, f, indent=4)

    print("✅ Saved labeled point cloud (scene.dump based, no scatter)")


if __name__ == "__main__":
    INPUT_URDF = "./1126_merge_robots/merge_fixed_joint_ur3_gripper.urdf"
    OUTPUT_DIR = "./output/ur3"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(INPUT_URDF))[0]

    parse_urdf_to_structure_json(INPUT_URDF, os.path.join(OUTPUT_DIR, f"{base}_structure.json"))
    generate_labeled_pointcloud_from_scene_dump(INPUT_URDF, OUTPUT_DIR, samples_per_mesh=2048)

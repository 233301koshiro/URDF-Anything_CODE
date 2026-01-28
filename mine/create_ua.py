import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import yourdfpy
import trimesh

# =============================================================================
# Part 1: 構造パラメータ(JSON)抽出
# =============================================================================
def parse_urdf_to_structure_json(urdf_path, output_json_path):
    print(f"📄 Parsing URDF for Structure JSON: {urdf_path}")
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints_data = []
    links_map = {}

    for link in root.findall('link'):
        link_name = link.get('name')
        links_map[link_name] = f"generic_part [SEG]"

    for joint in root.findall('joint'):
        joint_dict = {
            "id": joint.get('name'), "type": joint.get('type'),
            "parent": joint.find('parent').get('link'),
            "child": joint.find('child').get('link'),
            "origin": {"xyz": [0,0,0], "rpy": [0,0,0]}, "axis": [0,0,0]
        }
        
        origin = joint.find('origin')
        if origin is not None:
            if origin.get('xyz'): joint_dict["origin"]["xyz"] = [float(x) for x in origin.get('xyz').split()]
            if origin.get('rpy'): joint_dict["origin"]["rpy"] = [float(x) for x in origin.get('rpy').split()]
            
        axis = joint.find('axis')
        if axis is not None and axis.get('xyz'):
            joint_dict["axis"] = [float(x) for x in axis.get('xyz').split()]
            
        limit = joint.find('limit')
        if limit is not None:
            joint_dict["limit"] = {}
            if limit.get('lower'): joint_dict["limit"]['lower'] = float(limit.get('lower'))
            if limit.get('upper'): joint_dict["limit"]['upper'] = float(limit.get('upper'))

        joints_data.append(joint_dict)

    with open(output_json_path, 'w') as f:
        json.dump({"joints": joints_data, "links": links_map}, f, indent=4)
    print(f"✅ Saved Structure JSON: {output_json_path}")

# =============================================================================
# Part 2: 点群生成 (自動補正機能付き)
# =============================================================================
def generate_labeled_pointcloud(urdf_path, output_dir, samples_per_link=2048):
    print(f"☁️ Generating Point Cloud from: {urdf_path}")
    
    # 1. ロボット読み込み
    try:
        robot = yourdfpy.URDF.load(urdf_path, load_meshes=True, load_collision_meshes=False)
    except Exception as e:
        print(f"❌ URDF Load Error: {e}")
        return

    robot.update_cfg(configuration={j: 0.0 for j in robot.joint_map})
    
    # デバッグ用OBJ保存
    debug_mesh_path = os.path.join(output_dir, "debug_full_robot.obj")
    robot.scene.export(debug_mesh_path)
    
    root_frame = robot.scene.graph.base_frame
    all_points = []
    all_colors = []
    all_labels = []
    
    link_names = list(robot.link_map.keys())
    link_name_to_id = {name: i for i, name in enumerate(link_names)}
    
    print(f"   Scanning Scene Graph...")

    # --- 【自動判定ロジック】 ---
    # 変換行列の方向が正しいかテストする
    # テスト対象: 最初のリンク以外の適当なリンク (例: shoulder_link)
    test_link = None
    if len(link_names) > 1:
        test_link = link_names[1] # base以外
    else:
        test_link = link_names[0]

    # 正解の座標 (Kinematics)
    expected_matrix = robot.get_transform(test_link, root_frame) # World -> Link (Inverse) ?? No.
    # get_transform(to, from) -> Vector in From * M = Vector in To.
    # We want Local(Link) -> World. So to="world", from="link".
    expected_matrix = robot.get_transform(root_frame, test_link) 
    expected_pos = expected_matrix[:3, 3]

    print(f"   [Auto-Fix] Calibration using link: '{test_link}'")
    print(f"   [Auto-Fix] Expected Position (URDF): {expected_pos}")

    # 変換モードフラグ
    use_inverse_transform = False

    # シーン内からそのリンクのメッシュを探してテスト
    found_test_mesh = False
    for node in robot.scene.graph.nodes:
        if test_link in node: # 名前が含まれるノードを探す
            # 試しに transform を取得してみる
            try:
                # パターンA: frame_to=root (通常)
                t_a = robot.scene.graph.get(frame_to=root_frame, frame_from=node)
                if isinstance(t_a, tuple): t_a = t_a[0]
                pos_a = t_a[:3, 3]

                # パターンB: frame_to=node (逆)
                t_b = robot.scene.graph.get(frame_to=node, frame_from=root_frame)
                if isinstance(t_b, tuple): t_b = t_b[0]
                pos_b = t_b[:3, 3]

                dist_a = np.linalg.norm(pos_a - expected_pos)
                dist_b = np.linalg.norm(pos_b - expected_pos)

                # print(f"      Option A (Normal): {pos_a} (Err: {dist_a:.4f})")
                # print(f"      Option B (Inverse): {pos_b} (Err: {dist_b:.4f})")

                if dist_a < dist_b:
                    use_inverse_transform = False
                    # print("   -> Selected: Normal Transform")
                else:
                    use_inverse_transform = False
                    print("   -> Selected: Inverse Transform (Fixing orientation...)")
                
                found_test_mesh = True
                break
            except:
                continue
    
    if not found_test_mesh:
        print("   ⚠️ Calibration link not found in scene. Defaulting to Normal Transform.")

    # --- 本番処理 ---
    mesh_count = 0
    for node_name in robot.scene.graph.nodes:
        geom_name = robot.scene.graph[node_name][1]
        if geom_name is None: continue
        mesh_original = robot.scene.geometry.get(geom_name)
        if mesh_original is None: continue

        # 親リンク探し
        owner_link_name = None
        current_node = node_name
        for _ in range(50):
            if current_node in link_name_to_id:
                owner_link_name = current_node
                break
            parents = robot.scene.graph.transforms.parents.get(current_node)
            if parents is None: break
            current_node = parents
            
        if owner_link_name is None: continue
        
        current_label_id = link_name_to_id[owner_link_name]
        mesh_count += 1
        
        try:
            mesh_copy = mesh_original.copy()
            

            trans_res = robot.scene.graph.get(frame_to=root_frame, frame_from=node_name)

            if isinstance(trans_res, tuple) or isinstance(trans_res, list):
                global_transform = trans_res[0]
            else:
                global_transform = trans_res
            
            global_transform = np.array(global_transform)

            mesh_copy.apply_transform(global_transform)
            points, _ = trimesh.sample.sample_surface(mesh_copy, samples_per_link)
            
            all_points.append(points)
            all_colors.append(np.ones((len(points), 3)))
            all_labels.append(np.full(len(points), current_label_id))
            
        except Exception as e:
            print(f"   ⚠️ Error processing node {node_name}: {e}")

    if not all_points:
        print(f"❌ No valid points generated. (Mesh found count: {mesh_count})")
        return

    X = np.vstack(all_points)
    C = np.vstack(all_colors)
    L = np.concatenate(all_labels).reshape(-1, 1)

    # 正規化前のサイズチェック
    bbox = np.max(X, axis=0) - np.min(X, axis=0)
    print(f"   📏 Robot Dimensions (before norm): {bbox}")

    centroid = np.mean(X, axis=0)
    X -= centroid
    max_dist = np.max(np.linalg.norm(X, axis=1))
    if max_dist > 0:
        X /= max_dist
        print(f"   📏 Normalized: Scale=1.0/{max_dist:.4f}")

    final_data = np.hstack((X, C, L)).astype(np.float32)
    
    base_name = os.path.splitext(os.path.basename(urdf_path))[0]
    npy_path = os.path.join(output_dir, f"{base_name}_labeled.npy")
    label_map_path = os.path.join(output_dir, f"{base_name}_label_map.json")
    
    np.save(npy_path, final_data)
    with open(label_map_path, 'w') as f:
        json.dump(link_name_to_id, f, indent=4)

    print(f"✅ Saved Point Cloud: {npy_path}")
    print(f"✅ Saved Label Map:   {label_map_path}")
    print(f"👉 Now run check_coordinates.py to verify!")

if __name__ == "__main__":
    INPUT_URDF = "./1126_merge_robots/merge_fixed_joint_ur3_gripper.urdf"
    OUTPUT_DIR = "./output/ur3/"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(INPUT_URDF))[0]
    
    parse_urdf_to_structure_json(INPUT_URDF, os.path.join(OUTPUT_DIR, f"{base}_structure.json"))
    generate_labeled_pointcloud(INPUT_URDF, OUTPUT_DIR)
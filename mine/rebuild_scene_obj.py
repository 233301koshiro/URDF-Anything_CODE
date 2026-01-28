import os
import numpy as np
import yourdfpy
import trimesh

def to_mat(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    return np.array(x, dtype=float)

def main(urdf_path, out_dir="./output/ur3"):
    os.makedirs(out_dir, exist_ok=True)

    robot = yourdfpy.URDF.load(urdf_path, load_meshes=True, load_collision_meshes=False)
    robot.update_cfg(configuration={j: 0.0 for j in robot.joint_map})

    root = robot.scene.graph.base_frame
    print("[info] base_frame(root) =", root)

    # 1) yourdfpyがそのままexportしたOBJ（比較用）
    obj_ref = os.path.join(out_dir, "./output/ur3/debug_full_robot.obj")
    robot.scene.export(obj_ref)
    print("[OK] exported ref obj:", obj_ref)

    # 2) あなたの「nodeごとにtransformを取って適用」方式で再構成してexport
    recon = trimesh.Scene()

    # trimesh公式の「geometryを持つノード一覧」から回す（ここが重要）
    # ※あなたの for node in robot.scene.graph.nodes より安全
    nodes_geom = list(robot.scene.graph.nodes_geometry)
    print("[info] nodes_geometry count =", len(nodes_geom))

    for node in nodes_geom:
        geom_name = robot.scene.graph[node][1]
        mesh = robot.scene.geometry.get(geom_name)
        if mesh is None:
            continue

        # あなたが今固定している方式
        T = to_mat(robot.scene.graph.get(frame_to=root, frame_from=node))

        # add_geometryにtransformとして渡す（apply_transformで焼かない）
        recon.add_geometry(mesh.copy(), node_name=node, geom_name=f"{geom_name}_copy", transform=T)

    obj_recon = os.path.join(out_dir, "reconstructed_by_graph.obj")
    recon.export(obj_recon)
    print("[OK] exported reconstructed obj:", obj_recon)
    print("=> Choreonoidで debug_full_robot.obj と reconstructed_by_graph.obj を重ねて見比べてください")

if __name__ == "__main__":
    main("./1126_merge_robots/merge_fixed_joint_ur3_gripper.urdf", out_dir="./")

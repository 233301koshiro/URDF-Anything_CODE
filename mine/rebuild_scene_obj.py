import os
import numpy as np
import yourdfpy
import trimesh

def as_mesh_list(dumped):
    """
    trimeshのバージョン差を吸収して、dump結果を list[Trimesh] に揃える
    """
    if dumped is None:
        return []
    if isinstance(dumped, trimesh.Trimesh):
        return [dumped]
    if isinstance(dumped, list) or isinstance(dumped, tuple):
        meshes = []
        for x in dumped:
            meshes.extend(as_mesh_list(x))
        return meshes
    if isinstance(dumped, dict):
        # 古い版だと dict で返ることがある
        meshes = []
        for v in dumped.values():
            meshes.extend(as_mesh_list(v))
        return meshes
    # 想定外
    return []

def mesh_is_valid(m: trimesh.Trimesh) -> bool:
    if m is None:
        return False
    if not hasattr(m, "vertices") or len(m.vertices) == 0:
        return False
    if m.vertices.shape[1] != 3:
        return False
    if not np.isfinite(m.vertices).all():
        return False
    return True

def main(urdf_path, out_dir="./output"):
    os.makedirs(out_dir, exist_ok=True)

    robot = yourdfpy.URDF.load(urdf_path, load_meshes=True, load_collision_meshes=False)
    robot.update_cfg(configuration={j: 0.0 for j in robot.joint_map})

    scene = robot.scene

    print("[info] scene.geometry count =", len(scene.geometry))
    print("[info] graph.nodes count    =", len(list(scene.graph.nodes)))
    try:
        ng = list(scene.graph.nodes_geometry)
        print("[info] nodes_geometry count=", len(ng))
    except Exception as e:
        print("[warn] cannot read nodes_geometry:", e)

    # 参照（これはあなたの既存と同じ）
    ref_obj = os.path.join(out_dir, "debug_full_robot.obj")
    scene.export(ref_obj)
    print("[OK] exported:", ref_obj)

    # ★Scene全体をdumpしてワールド化メッシュ群を取得
    dumped = None
    try:
        dumped = scene.dump(concatenate=False)
    except TypeError:
        dumped = scene.dump()

    meshes = as_mesh_list(dumped)
    print("[info] dumped meshes raw count =", len(meshes))

    valid = []
    invalid = 0
    for m in meshes:
        if mesh_is_valid(m):
            valid.append(m)
        else:
            invalid += 1

    print("[info] valid meshes   =", len(valid))
    print("[info] invalid meshes =", invalid)

    if len(valid) == 0:
        # ここで落ちるなら dump が返してる中身がメッシュじゃない/空
        raise RuntimeError("No valid meshes dumped from scene.dump().")

    # Choreonoid互換性のためSTLで出す（OBJのパース地雷回避）
    big = trimesh.util.concatenate(valid)
    out_stl = os.path.join(out_dir, "reconstructed_scene_dump.stl")
    big.export(out_stl)
    print("[OK] exported:", out_stl)
    print("=> Choreonoidで debug_full_robot.obj と reconstructed_scene_dump.stl を比較してね")

if __name__ == "__main__":
    main("./1126_merge_robots/merge_fixed_joint_ur3_gripper.urdf", out_dir="./")

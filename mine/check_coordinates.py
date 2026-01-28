import numpy as np
import yourdfpy
import trimesh
import os
#メッシュから作った点群データがパーツごとにバラバラに配置されているため
#URDFの関節位置と照らし合わせて確認するスクリプト

def diagnose_dataset(npy_path, urdf_path):
    print(f"🔍 Diagnosing Point Cloud vs URDF Kinematics")
    print(f"   NPY:  {npy_path}")
    print(f"   URDF: {urdf_path}")

    # 1. 点群データの読み込み
    try:
        data = np.load(npy_path)
    except FileNotFoundError:
        print("❌ NPY file not found.")
        return

    points = data[:, :3] # XYZ
    labels = data[:, 6]  # Label ID
    
    # 2. ロボット(URDF)の読み込み
    robot = yourdfpy.URDF.load(urdf_path)
    robot.update_cfg(configuration={j: 0.0 for j in robot.joint_map})
    root_link = robot.scene.graph.base_frame

    # リンク名リスト取得
    link_names = list(robot.link_map.keys())
    
    print("\n" + "="*60)
    print(f"{'Link Name':<20} | {'Expected (URDF)':<20} | {'Actual (Point Cloud)':<20}")
    print("-" * 60)

    for i, link_name in enumerate(link_names):
        # --- A. 点群の重心 (Actual) ---
        # ラベルID i に対応する点を抽出
        part_points = points[labels == i]
        
        if len(part_points) == 0:
            actual_pos_str = "No Points"
        else:
            # 重心を計算
            centroid = np.mean(part_points, axis=0)
            # 正規化を戻すヒントを得るため、値のスケール感を見る
            actual_pos_str = f"[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"

        # --- B. URDFのリンク原点 (Expected) ---
        try:
            # ワールド座標系でのリンク位置を取得
            matrix = robot.get_transform(link_name, root_link)
            # matrixは4x4、平行移動成分は [0:3, 3]
            urdf_pos = matrix[0:3, 3]
            expected_pos_str = f"[{urdf_pos[0]:.3f}, {urdf_pos[1]:.3f}, {urdf_pos[2]:.3f}]"
        except:
            expected_pos_str = "Error"

        print(f"{link_name:<20} | {expected_pos_str:<20} | {actual_pos_str:<20}")
    print("="*60)
    
    # 全体のバウンディングボックスサイズを確認
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    size = bbox_max - bbox_min
    print(f"\n📏 Total Robot Size (XYZ): [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
    if np.max(size) > 10.0:
        print("⚠️  WARNING: The size is HUGE (>10). Likely unit mismatch (mm vs m).")
    elif np.max(size) < 0.05:
        print("⚠️  WARNING: The size is TINY (<0.05). Check scale.")
    else:
        print("✅ Size seems reasonable for a robot (meters).")

if __name__ == "__main__":
    # ここに「今回生成されたNPYファイル」と「元のURDF」を指定してください
    INPUT_NPY = "./output/ur3/merge_fixed_joint_ur3_gripper_labeled.npy"
    INPUT_URDF = "./1126_merge_robots/merge_fixed_joint_ur3_gripper.urdf"
    
    diagnose_dataset(INPUT_NPY, INPUT_URDF)
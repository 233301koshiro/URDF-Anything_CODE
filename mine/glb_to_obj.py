import trimesh
import numpy as np
import os

def glb_to_urdf_anything_input(glb_path, export_obj_path, npy_path, target_count=8192):
    """
    GLBファイルを読み込み、URDF-Anything学習/推論用の点群データ(.npy)を作成する。
    
    Args:
        glb_path (str): 入力GLBファイルのパス
        export_obj_path (str): 確認用OBJファイルの保存先
        npy_path (str): 出力NPYファイルの保存先
        target_count (int): 点群の点数 (デフォルト8192)
    """
    print(f"🔄 Processing: {glb_path}")

    # 1. GLB読み込み
    mesh = trimesh.load(glb_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # 2. 回転補正 (Tripoのモデルが寝ている場合のみ有効化。通常は不要な場合も多いが、一応入れる)
    # ※ モデルが直立しているならこのブロックはコメントアウトしてもOK
    print("   -> Applying Rotation (X-axis -90 deg for Z-up correction)")
    matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    mesh.apply_transform(matrix)
    

    # 3. 原点合わせ (Centering)
    mesh.apply_translation(-mesh.centroid)

    # 4. 正規化 (Normalization) - 単位球(半径1)に収める
    # これがニューラルネット入力には必須
    max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if max_dist > 0:
        scale = 1.0 / max_dist
        mesh.apply_scale(scale)
        print(f"   -> Applied Scaling: {scale:.4f} (Original Size: {max_dist:.4f})")

    # 5. 確認用OBJ保存 (この時点での形状がAIに入力される)
    os.makedirs(os.path.dirname(export_obj_path), exist_ok=True)
    mesh.export(export_obj_path)
    print(f"   -> Saved Debug OBJ: {export_obj_path}")

    # 6. ハイブリッド・サンプリング
    # (A) 表面サンプリング (70%)
    count_surface = int(target_count * 0.7)
    points_surface, face_indices = trimesh.sample.sample_surface(mesh, count_surface)
    
    # 色情報の取得 (表面サンプリング分)
    colors_surface = np.ones((count_surface, 3)) # デフォルトは白
    
    # テクスチャ/頂点カラーがある場合、それを取得する努力をする
    if hasattr(mesh.visual, 'to_color'):
        try:
            # UVマッピングなどから色を取得できる場合
            visual_color = mesh.visual.to_color()
            if hasattr(visual_color, 'face_colors'):
                # face_indicesを使ってサンプリング点の色を取得
                colors_surface = visual_color.face_colors[face_indices][:, :3]
                # 0-255なら0-1に正規化
                if colors_surface.max() > 1.1:
                    colors_surface = colors_surface / 255.0
        except Exception as e:
            print(f"   -> Warning: Color extraction failed ({e}). Using white.")

    # (B) メッシュの頂点を混ぜる (30%) - 形状のエッジを保つため
    count_verts = target_count - count_surface
    verts = mesh.vertices
    if len(verts) > 0:
        # 頂点数が足りない場合は重複許可(replace=True)、足りるならFalse
        replace = len(verts) < count_verts
        indices = np.random.choice(len(verts), count_verts, replace=replace)
        points_verts = verts[indices]
        
        # 頂点の色取得
        colors_verts = np.ones((count_verts, 3))
        if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) == len(verts):
             colors_verts = mesh.visual.vertex_colors[indices][:, :3]
             if colors_verts.max() > 1.1:
                    colors_verts = colors_verts / 255.0
    else:
        points_verts = np.empty((0, 3))
        colors_verts = np.empty((0, 3))

    # 7. 結合 (Merge)
    points = np.vstack((points_surface, points_verts))
    colors = np.vstack((colors_surface, colors_verts))
    
    # 点数が厳密に target_count になるように調整 (稀にズレるため)
    if len(points) > target_count:
        points = points[:target_count]
        colors = colors[:target_count]
    elif len(points) < target_count:
        # 足りない分は最後の点をコピーして埋める
        pad_size = target_count - len(points)
        points = np.vstack((points, points[-pad_size:]))
        colors = np.vstack((colors, colors[-pad_size:]))

    # 8. 保存 (XYZ + RGB = 6次元)
    # shape: (8192, 6)
    point_cloud_data = np.hstack((points, colors)).astype(np.float32)
    
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, point_cloud_data)
    print(f"✅ Saved NPY: {npy_path} (Shape: {point_cloud_data.shape})")

# --- 実行 ---
if __name__ == "__main__":
    # パスは適宜書き換えてください
    INPUT_GLB = "./output/arm/arm.glb"
    DEBUG_OBJ = "./output/arm/check_me.obj"
    OUTPUT_NPY = "./output/arm/arm.npy"
    
    # ファイルがあるか確認
    if os.path.exists(INPUT_GLB):
        glb_to_urdf_anything_input(INPUT_GLB, DEBUG_OBJ, OUTPUT_NPY)
    else:
        print(f"Error: Input file not found: {INPUT_GLB}")
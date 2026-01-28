import trimesh
import numpy as np
from scipy.spatial import cKDTree
import os

def split_mesh_using_labels(mesh_path, npy_path, output_dir):
    print(f"📂 Mesh: {mesh_path}")
    print(f"📂 Labels: {npy_path}")

    # 1. データの読み込み
    try:
        # 位置合わせ済みのメッシュを読む
        mesh = trimesh.load(mesh_path)
        
        # ラベル付き点群を読む [x, y, z, r, g, b, label]
        data = np.load(npy_path)
        points = data[:, :3]
        labels = data[:, 6] # 0 or 1
        
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return

    print(f"   Mesh Vertices: {len(mesh.vertices)}")
    print(f"   Labeled Points: {len(points)}")

    # 2. k-NNで「点群のラベル」を「メッシュの頂点」にコピーする
    print("🔄 k-NNを実行中 (点群 -> メッシュ頂点)...")
    
    # 点群で検索ツリーを作る
    tree = cKDTree(points)
    
    # メッシュの各頂点について、一番近い点群の点を探す
    distances, indices = tree.query(mesh.vertices, k=1)
    
    # 近かった点のラベルを採用
    vertex_labels = labels[indices]

    # 3. 面 (Face) 単位での分割
    # ポリゴン（三角形）は3つの頂点でできているので、頂点のラベルの多数決をとる
    # (例: 3つのうち2つが「頭」なら、その面は「頭」)
    face_labels_mean = vertex_labels[mesh.faces].mean(axis=1)
    
    # 0.5より大きければ Head(1), 以下なら Body(0)
    head_faces_mask = face_labels_mean > 0.5
    body_faces_mask = ~head_faces_mask # 反転

    print(f"   Head Faces: {np.sum(head_faces_mask)}")
    print(f"   Body Faces: {np.sum(body_faces_mask)}")

    # 4. サブメッシュの作成と保存
    # process=False にすると、座標を勝手に動かさず、元の位置をキープします（重要）
    head_mesh = mesh.submesh([head_faces_mask], append=True)
    body_mesh = mesh.submesh([body_faces_mask], append=True)

    os.makedirs(output_dir, exist_ok=True)
    
    # 保存
    head_out = os.path.join(output_dir, "link_head.obj")
    body_out = os.path.join(output_dir, "link_body.obj")
    
    head_mesh.export(head_out)
    body_mesh.export(body_out)

    print("\n✅ 分割成功！")
    print(f"   -> {head_out}")
    print(f"   -> {body_out}")
    print("   (Blenderなどでこの2つを同時に開いてみてください)")

if __name__ == "__main__":
    # 【注意】必ず「位置合わせ済み」のOBJファイルを使うこと！
    INPUT_MESH = "./output/check_me.obj" 
    
    # ラベル付き点群
    INPUT_NPY = "./output/snowman_labeled.npy"
    
    # 出力先
    OUTPUT_DIR = "./output"

    split_mesh_using_labels(INPUT_MESH, INPUT_NPY, OUTPUT_DIR)
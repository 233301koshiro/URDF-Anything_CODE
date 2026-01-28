import numpy as np
import matplotlib.pyplot as plt

def visualize_labeled_pointcloud(npy_path):
    print(f"📂 Reading: {npy_path}")
    
    # データ読み込み [x, y, z, r, g, b, label]
    try:
        data = np.load(npy_path)
    except FileNotFoundError:
        print("❌ ファイルが見つかりません。パスを確認してください。")
        return

    # 座標とラベルを取得
    points = data[:, :3]      # [x, y, z]
    colors_original = data[:, 3:6] 
    labels = data[:, 6]       # 0, 1, 2, ...
    
    # ラベルの種類（パーツ数）を確認
    unique_labels = np.unique(labels)
    num_parts = len(unique_labels)
    print(f"   -> 検出されたパーツ数: {num_parts} (Labels: {unique_labels})")

    # --- 可視化設定 ---
    fig = plt.figure(figsize=(14, 7))
    
    # ==========================================
    # 1. 元の色で表示 (Original Colors)
    # ==========================================
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Colors")
    
    # ほぼ白の場合は見やすくグレーにする
    if np.mean(colors_original) > 0.95: 
        c_show = 'gray'
    else:
        c_show = colors_original

    # Z-upデータなので、そのまま x, y, z でプロットします
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=c_show, alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Height)')
    # 見やすい角度に調整 (Elev=高さ角度, Azim=回転角度)
    ax1.view_init(elev=20, azim=30) 

    # ==========================================
    # 2. セグメンテーション結果 (Segmentation Labels)
    # ==========================================
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Segmentation Result ({num_parts} Parts)")
    
    # --- 【ここが修正ポイント】パーツごとに自動で色を作る ---
    colors_seg = np.zeros_like(points)
    
    # Matplotlibのカラーマップ 'tab10' (10色のパレット) を使用
    # パーツが10個以上ある場合は 'tab20' に変えてください
    cmap = plt.get_cmap("tab10") 
    
    for i, label in enumerate(unique_labels):
        # ラベル番号に基づいて色を取得 (RGBAのうちRGBだけ使う)
        # int(label) % 10 にすることで、ラベルが10を超えても色が循環してエラーにならない
        color = cmap(int(label) % 10)[:3]
        
        # そのラベルを持つ点だけに色を塗る
        colors_seg[labels == label] = color
        
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors_seg, alpha=0.8)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z (Height)')
    ax2.view_init(elev=20, azim=30)

    # 保存
    output_img = npy_path.replace(".npy", "_vis.png")
    plt.savefig(output_img, dpi=150)
    print(f"✅ 確認画像を出力しました: {output_img}")
    plt.show()

if __name__ == "__main__":
    # 複数分割したファイルパス ( _multi.npy ) を指定してください
    input_npy = "./output/ur3/merge_fixed_joint_ur3_gripper_labeled.npy"
    
    visualize_labeled_pointcloud(input_npy)
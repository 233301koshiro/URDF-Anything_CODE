import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def find_boundaries_and_multi_segment(npy_path, prominence_ratio=0.15, min_dist_bins=5):
    """
    Args:
        npy_path: 点群データのパス
        prominence_ratio: くびれの深さ判定 (0.0~1.0)。大きいほど深い谷だけ拾う。
        min_dist_bins: くびれ同士の最小間隔 (ビン数)。近すぎる谷を無視する。
    """
    # 1. 点群データの読み込み
    data = np.load(npy_path)
    points = data[:, :3] # [x, y, z]
    
    # 高さ (Z軸)
    height_coords = points[:, 2] 
    
    # 2. ヒストグラムを作成
    # binsを少し増やして分解能を上げます (50 -> 100)
    bins_count = 100
    hist, bin_edges = np.histogram(height_coords, bins=bins_count)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 3. 「くびれ（谷）」をすべて見つける
    # 山を探す関数なので、データを反転させて谷を探します
    inverted_hist = np.max(hist) - hist
    
    # パラメータ設定
    # prominence: 周りよりどれくらい深いか
    # distance: 隣のピークとどれくらい離れているか (ビン単位)
    peaks, properties = find_peaks(
        inverted_hist, 
        prominence=np.max(inverted_hist) * prominence_ratio,
        distance=min_dist_bins
    )
    
    # くびれのZ座標リスト
    boundaries = bin_centers[peaks]
    boundaries.sort() # 下から順に並べる
    
    print(f"✅ 検出されたくびれ数: {len(boundaries)}")
    for i, b in enumerate(boundaries):
        print(f"   - Boundary {i+1}: Z = {b:.4f}")

    # --- 可視化 ---
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, hist, label='Point Density')
    
    # 検出されたすべての境界線を引く
    for b in boundaries:
        plt.axvline(x=b, color='r', linestyle='--', alpha=0.7)
        
    plt.title(f"Z-axis Point Density ({len(boundaries)} cuts found)")
    plt.xlabel("Height (Z)")
    plt.ylabel("Number of Points")
    plt.legend()
    plt.savefig("debug_histogram_multi.png")
    print("📊 分布図を 'debug_histogram_multi.png' に保存しました。")
    # --------------

    # 4. ラベル付け (Multi-segmentation)
    # np.digitize を使うと、境界線リストを使って一発で 0, 1, 2... に振り分けてくれます
    # boundaries = [z1, z2] の場合:
    #   z < z1  -> 0
    #   z1 <= z < z2 -> 1
    #   z2 <= z -> 2
    labels = np.digitize(height_coords, boundaries)
    
    # 5. 保存
    # [x, y, z, r, g, b, label]
    labeled_data = np.hstack((data, labels.reshape(-1, 1)))
    
    output_path = npy_path.replace(".npy", "_labeled_multi.npy")
    np.save(output_path, labeled_data)
    print(f"💾 ラベル付き点群を保存しました: {output_path}")
    print(f"   -> 合計パーツ数: {len(boundaries) + 1}")
    
    return labeled_data, boundaries

if __name__ == "__main__":
    input_npy = "./output/arm/arm.npy"
    
    try:
        # パラメータ調整のコツ:
        # - prominence_ratio: くびれを逃すなら下げる (0.1)、ゴミを拾うなら上げる (0.2)
        # - min_dist_bins: 近すぎる2本線が出るなら値を大きくする (5 -> 10)
        find_boundaries_and_multi_segment(input_npy, prominence_ratio=0.15, min_dist_bins=8)
        
    except FileNotFoundError:
        print("❌ ファイルが見つかりません。")
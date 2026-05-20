# CHANGES

このリポジトリで行った最近の整理の要約です。

## 概要
- 自作スクリプト類を `mine/scripts/` 以下へ用途別に整理しました。
- 元来リポジトリに含まれていたファイルはルートに戻しました（例: `eval.py`, `eval.sh` など）。

## 関連コミット（重要な修正）
- `9eb6b27` fix(data): normalize RGB color values to [0,1] range for rrbot_test point clouds
  - **根本原因特定**: rrbot_test の point clouds は RGB が [0, 255] range だが、モデルは [0, 1] を期待
  - 自動正規化を実装し、JSON 生成破損を解決
- `497f031` chore(repo): organize custom scripts under mine
  - スクリプトを `mine/scripts/` 配下へ移動する大規模な整理（rename）。
- `0ef89a1` chore(repo): restore original root files (restore upstream files to root)
  - 元々ルートにあったファイル群をルートへ戻しました。
- `0d003f9` fix(data): apply URDF transforms in pointcloud generator
  - `generate_pointcloud_from_urdf.py` の修正（URDF変換の適用）。

## JSON 破損問題の根本原因

### 問題の発見過程
1. `output_light`（2026-05-13）では retry なしで有効な JSON が生成されていた
2. `output_custom/rrbot_test` では無効な JSON が生成されていた
3. 推論パイプラインは 2026-05-13～05-20 で実質変更なし（デバッグ出力と retry ロジックのみ追加）

### 根本原因の特定
- **urdf point clouds**: RGB 値は [0.0, 0.0, 0.0] など [0, 1] 範囲（正規化済み）
- **rrbot_test point clouds**: RGB 値は [242, 48, 48] など [0, 255] 範囲（正規化なし）
- モデル入力: `[x, y, z, R, G, B]` という 6D ベクトル
- モデルは [0, 1] の色を期待していたため、[0, 255] の入力で出力がおかしくなった

### 修正方法
`utils/new_dataset.py` の `_load_point_txt()` メソッドで自動正規化:
- いずれかの RGB 値が 1.0 を超えていたら、すべての RGB を 255.0 で割る
- これにより rrbot_test も他のデータセットと統一された形式で処理される
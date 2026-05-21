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

## 追記: JSON破損・seg_count・retry 挙動の調査と改善（2026-05-21/22）

### 実施した改善
- `train_lightning.py`
  - `retry_even_if_json_valid` オプションを追加（有効時、JSONが有効でも2回目生成を実施）。
  - 1回目/2回目の状態を詳細ログ化:
    - `TEST_STEP_RETRY_POLICY`
    - `TEST_STEP_RETRY_INITIAL_STATS`
    - `TEST_STEP_RETRY_POST_STATS`
    - `TEST_STEP_RETRY_*_JSON`
  - 出力JSONに `retry_trace` を保存し、前後比較を可能化。
- `mine/scripts/eval/eval_custom.sh`
  - `RETRY_EVEN_IF_JSON_VALID` 環境変数を `eval.py` に伝搬。
  - `open3d` 未導入時に自動インストールする処理を追加（カスタム評価の依存解決）。
- `utils/new_dataset.py`
  - RGB正規化に加え、座標正規化を追加（`point_normalize=True` 既定）。
  - `pc_normalize`（重心原点化 + 単位球スケーリング）を `__getitem__` に適用。
- `train_lightning.py` / `DataArguments`
  - `point_normalize` 引数を追加し、train/val/test の全データセット構築時に反映。

### 実験結果（最新の強制retry実験）
- ログ: `mine/scripts/eval/logs/eval_custom_20260521_051320.log`
- 設定: `retry_even_if_json_valid=True`
- 観測:
  - 初回: `initial_json_valid=True`, `initial_seg_count=2`
  - 2回目: `retry_json_valid=False`, `retry_seg_count=2`
  - 期待していた `seg_count 2→5` は再現せず、今回は逆に JSON が壊れる結果。

### 現時点の整理
- 色正規化は「壊れた JSON を減らす」方向で有効。
- ただし、強制retryは常に有利ではなく、JSONを悪化させるケースがある。
- 座標正規化を追加したため、元データセットとの分布乖離（位置・スケール差）は縮小。
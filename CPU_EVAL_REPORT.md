# CPU 実行レポート

作成日: 2026-05-06

## 1. 目的

このリポジトリの CPU 環境で、`eval.sh` による推論・URDF生成の流れを動かし、
「入力データが何で、どこに出力され、結果がどんな内容か」を確認した。

## 2. 実行した流れ

1. Docker イメージを CPU 向けにビルド
2. 依存関係の不足を解消
   - `trimesh` を追加
   - `open3d` などの実行時依存を導入
3. PyTorch Lightning / Transformers の古い API 差分を吸収
4. `eval.sh` 側で `--limit_test_batches 1` を追加し、1バッチだけ実行できるようにした
5. `eval.py` 側で `limit_test_batches` を `Trainer` に渡すように修正
6. CPU 推論を実行し、出力ファイルを確認した

## 3. 実行環境

- OS: Linux
- 実行方式: Docker CPU 実行
- コンテナイメージ: `urdf_any_cpu:latest`
- 推論スクリプト: `bash docker_run_eval.sh`
- 評価モード: CPU-only
- バッチ制限: 1

## 4. 重要な修正点

### 4.1 `trimesh` の追加

mesh 生成で `ModuleNotFoundError: No module named 'trimesh'` が出ていたため、
`environment.yml` に `trimesh==4.4.9` を追加した。

### 4.2 `limit_test_batches` 対応

`eval.sh` に `--limit_test_batches 1` を追加しただけでは、`HfArgumentParser` が未認識で `ValueError` になった。
そこで、`train_lightning.py` の `TrainingArguments` に `limit_test_batches` を追加し、
`eval.py` の `Trainer(...)` に渡すように修正した。

### 4.3 早期終了の解除

`test_step()` にあった 1 バッチ後の強制終了を外し、
必要な後処理まで進められるようにした。

## 5. 実行結果の格納場所

主な出力先は次の通り。

- ルート出力ディレクトリ: `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/`
- 評価ログ: `eval_run_retry.log`
- エラー監視ログ: `eval_errors.log`

### 5.1 出力ディレクトリの中身

- `test/articulation_params/`
  - 予測・GT の関節パラメータ JSON
- `test/seg_visualization/`
  - セグメンテーション可視化の PLY
- `test/reconstructed/<sample_id>/pred/`
  - 予測された URDF とメッシュ
- `test/reconstructed/<sample_id>/gt/`
  - GT 側の URDF とメッシュ

## 6. 生成された結果の内容

確認したサンプル `100071_0` では、以下が生成されていた。

### 6.1 入力

- Point Cloud: `datasets/urdf/point_clouds/100071/100071_0.ply`
- Question: `datasets/urdf/json_questions/100071/100071_0.json`

このサンプルは USB 系の articulated object で、
「2部品のセグメンテーション結果と joint パラメータを JSON で出力する」タイプの入力だった。

### 6.2 生成結果

- 予測 URDF: `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/test/reconstructed/100071_0/pred/mobility.urdf`
- 予測メッシュ:
  - `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/test/reconstructed/100071_0/pred/meshes/link_0.obj`
  - `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/test/reconstructed/100071_0/pred/meshes/link_0.ply`
- セグメンテーション可視化:
  - `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/test/seg_visualization/100071_0_pred_fused.ply`
  - `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/test/seg_visualization/100071_0_gt_fused.ply`
- 関節パラメータ JSON:
  - `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/test/articulation_params/100071_0_pred.json`
  - `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/test/articulation_params/100071_0_gt.json`

## 7. 生成物の確認結果

### 7.1 ファイル数・容量

- 総ファイル数: 364
- 総容量: 約 18MB

内訳:
- JSON: 84
- OBJ: 84
- PLY: 140
- URDF: 56

### 7.2 破損チェック

- JSON は `sample_id`, `normalize`, `articulation` を含む正常な構造
- URDF は `<robot>`, `<link>`, `<joint>` を含む正常な XML
- PLY はヘッダが正しく、頂点と RGB 情報を持つ
- OBJ も正常に生成されている

## 8. CPU 実行で分かったこと

- 1サンプル推論だけでもかなり重い
- 以前の計測では、1 batch の推論に約 3589 秒かかった
- そのため、全 3935 サンプルを CPU で回すのは非常に時間がかかる
- まずは `--limit_test_batches 1` で動作確認する流れが妥当

## 9. まとめ

CPU 実行の流れは以下の通りだった。

- Docker CPU イメージを作成
- 依存関係と古い API 差分を修正
- 1バッチ評価で推論を実行
- `output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test/` に URDF / メッシュ / JSON / 可視化を保存
- 生成物は空ではなく、破損も見られなかった

結論として、CPU 実行の推論パイプラインは動作し、結果は適切に格納されている。

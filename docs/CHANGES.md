# CHANGES

このリポジトリで行った最近の整理の要約です。

## 概要
- 自作スクリプト類を `mine/scripts/` 以下へ用途別に整理しました。
- 元来リポジトリに含まれていたファイルはルートに戻しました（例: `eval.py`, `eval.sh` など）。

## 関連コミット
- `497f031` chore(repo): organize custom scripts under mine
  - スクリプトを `mine/scripts/` 配下へ移動する大規模な整理（rename）。
- `0ef89a1` chore(repo): restore original root files (restore upstream files to root)
  - 元々ルートにあったファイル群をルートへ戻しました。
- `0d003f9` fix(data): apply URDF transforms in pointcloud generator
  - `generate_pointcloud_from_urdf.py` の修正（URDF変換の適用）。

## ルール

## 移動したユーザー作成のドキュメント

## 追記: 最近の推論不調の調査
- 最近の `rrbot_test` 推論不調は、`UA.py` 自体の単純な破損というより、**原データと自作データの乖離**に起因する可能性が高いと整理しました。
- 原データ側は、`point_cloud`・`answer.links`・`answer.joints` が学習時の構造に近く、`[SEG]` を含む自然な JSON 例になっていました。
- 一方 `rrbot_test` 側は、部品名が `handle` に潰れやすく、`answer.joints` も空になりやすいため、モデルが JSON 生成や `[SEG]` 出力を続けにくい状態でした。
- この差を埋めるため、`mine/scripts/data/urdf_to_eval_json.py` に `rrbot_test` 向けの具体例（`--example rrbot_test`）を追加し、`matching_part_map.py` で part map を作ってから JSON を作る流れを明示しました。
必要であれば、このファイルにさらに詳細（元のコミットハッシュ一覧や移動前後のパス）を追加します。
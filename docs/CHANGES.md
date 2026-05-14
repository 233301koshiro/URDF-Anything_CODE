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
- ユーザーが新たに作成・管理するファイル（例: データ生成スクリプトや個人設定）は `mine/` 以下に置きます。
- 元来のリポジトリファイル（アップストリーム由来の実行スクリプトやREADME等）はルートに残します。

## 移動したユーザー作成のドキュメント
- `CPU_EVAL_REPORT.md` -> `mine/docs/CPU_EVAL_REPORT.md` (added by 233301koshiro)
- `CPU_PROGRESS_GUIDE.md` -> `mine/docs/CPU_PROGRESS_GUIDE.md` (added by 233301koshiro)

必要であれば、このファイルにさらに詳細（元のコミットハッシュ一覧や移動前後のパス）を追加します。
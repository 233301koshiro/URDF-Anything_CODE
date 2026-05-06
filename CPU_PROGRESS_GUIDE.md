# LLaVA/LISA CPU 実行 - 進捗可視化ガイド

## 概要

CPU 環境での実行中に進捗が止まっているのか実行中なのかを確認できるように、詳細なログ出力と進捗表示機能を追加しました。

## 主な改善点

### 1. **詳細なログ出力**
すべての実行スクリプトで以下の情報を表示します：
- 初期化ステップの進捗
- モデル読み込み状況
- データセット準備状況
- メモリ使用状況
- 経過時間

### 2. **段階的な進捗表示**
各スクリプトで以下のようなステップ表示を行います：
```
[1/6] Parsing arguments...
[2/6] Initializing model...
[3/6] Loading checkpoint...
[4/6] Preparing dataset...
[5/6] Setting up training...
[6/6] Starting training loop...
```

### 3. **リアルタイムログ記録**
標準出力をログファイルに同時に記録します：
```bash
2>&1 | tee "${LOG_FILE}"
```

## 実行方法

### 方法1: 統合テストスクリプト（推奨）

```bash
bash test_llava_cpu.sh
```

このスクリプトは以下を実行します：
- CUDA 可用性確認
- PyTorch インストール確認  
- LLaVA モデル読み込みテスト
- 簡単な推論テスト
- 進捗をリアルタイム表示

ログは自動的に `logs/test.log` に保存されます。

### 方法2: 評価スクリプト

```bash
bash eval.sh
```

改善された点：
- 6段階のセットアップ進捗表示
- チェックポイント読み込み状況
- データセット準備の詳細情報
- 評価完了時の統計情報

ログファイル: `logs/eval_YYYYMMDD_HHMMSS.log`

### 方法3: トレーニングスクリプト

```bash
bash run_train_cpu.sh  # 新しい CPU 最適化版
# または
bash run_train.sh      # オリジナル版
```

改善された点：
- トレーニング開始前の全セットアップ状況表示
- 各エポックの進捗バー（PyTorch Lightning 組み込み）
- TensorBoard ロギング
- 完了時間の表示

ログファイル: `logs/train_MMDD_HHMM.log`

## ログの見方

### テストスクリプルの出力例

```
=======================================================================
  LLaVA CPU Verification Test
=======================================================================

[Step 1] Checking CUDA availability...
  CUDA Available: False
  CUDA Device Count: 0
✓ CUDA check completed

[Step 2] Setting up workspace...
  Workspace: /home/irsl/URDF-Anything_CODE_CPU
  Model path: /home/irsl/URDF-Anything_CODE_CPU/checkpoints/ShapeLLM_7B_gapartnet_v1.0
✓ Workspace setup completed

[Step 3] Verifying model checkpoint...
✓ Model checkpoint found
  Checkpoint files: 47 items
    - config.json
    - model.safetensors
    ...
```

### トレーニングスクリプルの出力例

```
=========================================================================
  LLaVA/LISA Training Script
=========================================================================
Current time: 0501_1430
Log directory: ./logs
CPU-optimized training mode

[1/6] Parsing arguments...
  Model: ./checkpoints/ShapeLLM_7B_gapartnet_v1.0
  Output dir: ./output/checkpoints/...

[2/6] Initializing random seed and tokenizer...
  Tokenizer vocab size: 32000
  Max length: 2048

[3/6] Loading LLaVA/LISA model...
  Model parameters: 7,123,456,789

[4/6] Preparing training dataset...
  Dataset configured and ready

[5/6] Setting up training callbacks...
  Checkpoint callback enabled
  TensorBoard logger: ./output/logs

[6/6] Initializing PyTorch Lightning Trainer...
  Configuration:
    - Accelerator: CPU (fully compatible, may be slower)
    - Precision: float32 (optimal for CPU)
    - Devices: 1 CPU
    - Gradient accumulation: 10

=========================================================================
  Starting Training Loop
=========================================================================
```

## ログファイルの場所

すべてのログは `logs/` ディレクトリに保存されます：

```
logs/
  ├── test.log                    # テスト実行ログ
  ├── eval_20260501_143000.log   # 評価実行ログ
  ├── train_0501_1430.log        # トレーニング実行ログ
  └── ...
```

## CPU 実行中の注意点

### 進捗が遅い場合

CPU での実行は GPU に比べて以下の通り遅くなります：

| 操作 | GPU（参考） | CPU |
|------|-----------|-----|
| モデル読み込み | 30秒 | 2-5分 |
| 単一トークン推論 | 0.1秒 | 1-3秒 |
| エポック学習 | 5分 | 30-60分 |

**スクリプトが止まっているのか実行中かを判断する方法：**

1. **ログを確認**
```bash
tail -f logs/train_*.log
```

2. **CPU 使用率を確認**
```bash
top  # または htop
```
CPU 使用率が > 50% なら実行中

3. **プロセスを確認**
```bash
ps aux | grep python
```
python プロセスが存在していれば実行中

### メモリ使用状況

CPU 実行では、通常より多くの RAM を使用します。

- **推奨**: 32GB 以上の RAM
- **最小**: 16GB RAM + スワップ 16GB

メモリ使用状況の確認：
```bash
free -h
```

## トラブルシューティング

### エラーが発生したが、ログ出力がない

```bash
# スクリプトの実行結果を確認
bash test_llava_cpu.sh 2>&1 | head -100
```

### Ctrl+C で中断した場合

実行中のスクリプトは以下のように安全に中断できます：
```
^C  # Ctrl+C を押す
```

最後に実行した部分まではログに記録されます。

### ログファイルが大きくなった

```bash
# 古いログを削除
rm logs/train_*_*.log

# または圧縮
gzip logs/train_*.log
```

## 更新内容の詳細

### 修正したファイル

1. **test_llava_cpu.py** - 詳細な進捗表示付きテストスクリプト
2. **eval.py** - 6段階の進捗表示を追加
3. **train_lightning.py** - 詳細な初期化ログと完了レポートを追加
4. **eval.sh** - ログ記録機能を追加
5. **run_train_cpu.sh** - CPU 最適化版トレーニングスクリプト（新規）
6. **test_llava_cpu.sh** - テスト用シェルスクリプト（新規）
7. **utils/progress.py** - 進捗可視化ユーティリティ（新規）

## CPU 実行の CPU 互換性確認

すべてのスクリプトは以下の CPU 互換チェックを行います：

- ✅ `.cuda()` の代わりに `.to(device)` を使用
- ✅ `torch.float32` を使用（float16/bfloat16 は使用しない）
- ✅ CUDA-only オプション（BitsAndBytes, Flash Attention など）を無効化
- ✅ device_map を CPU に設定
- ✅ Triton/CUDA Attention の代わりに PyTorch 標準 Attention を使用

## 次のステップ

進捗確認後：
1. モデルの動作確認: `bash test_llava_cpu.sh`
2. 評価実行: `bash eval.sh`
3. トレーニング開始: `bash run_train_cpu.sh`

各スクリプルはログを出力し、進捗が明確に可視化されます。

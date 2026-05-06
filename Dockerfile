FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100

# 必要なビルドツールとシステムライブラリをまとめてインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    pkg-config \
    python3-dev \
    libopenblas-dev \
    libblas-dev \
    gfortran \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/URDF-Anything

# environment.yml をコンテナにコピー
COPY environment.yml /tmp/environment.yml

# 1. pipのアップグレードと、依存関係のフィルタリングスクリプトの実行
RUN python -m pip install --upgrade pip "setuptools<70.0.0" wheel pyyaml cython \
 && python - <<'PY'
from pathlib import Path
import yaml

env_path = Path('/tmp/environment.yml')
if env_path.exists():
    env = yaml.safe_load(env_path.read_text())
    pip_deps = [dep for d in env.get('dependencies', []) if isinstance(d, dict) and 'pip' in d for dep in d['pip']]
else:
    pip_deps = []

# CUDA専用パッケージや、エラーの原因になるものを除外（llavaを追加）
skip_exact = {
    'bitsandbytes==0.41.0',
    'pointnet2-ops==3.0.0',
    'torch==2.0.1',
    'torch-scatter==2.1.2+pt20cu117',
    'torchaudio==2.2.0+cu118',
    'torchvision==0.15.2',
    'triton==2.0.0',
    'llava==1.1.3',
}

filtered = []
for raw in pip_deps:
    item = raw.strip()
    if not item:
        continue
    # パッケージ名だけを抽出して判定
    name = item.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].lower()
    if name.startswith('nvidia-') or item in skip_exact:
        continue
    filtered.append(item)

# 出力ファイル名を "requirements.cpu.txt" (ドット) に統一
Path('/tmp/requirements.cpu.txt').write_text('\n'.join(filtered) + '\n')
PY

# 2. PyTorch関連（CPU専用版）を明示的にインストール
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# 3. フィルタリングされた残りのパッケージをインストール
RUN pip install -r /tmp/requirements.cpu.txt

# 3.5. Additional CPU-specific packages
RUN pip install trimesh

# 4. pointnet2-ops をソースからビルド (CPU互換モード)
RUN git clone https://github.com/erikwijmans/PointNet2_PyTorch.git /tmp/PointNet2_PyTorch \
 && pip install -e /tmp/PointNet2_PyTorch \
 && rm -rf /tmp/PointNet2_PyTorch

# ソースコードをすべてコピー
COPY . .

# PYTHONPATHの未定義警告を防ぐために :- を追加
ENV PYTHONPATH=/workspace/URDF-Anything:${PYTHONPATH:-} \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TOKENIZERS_PARALLELISM=false

CMD ["/bin/bash"]
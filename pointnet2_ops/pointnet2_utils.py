"""CPU fallback implementation for pointnet2_ops.pointnet2_utils.

Provides minimal APIs used by this repository:
- furthest_point_sample(data, npoint)
- gather_operation(features, idx)

Shapes:
- data: (B, N, 3)
- features: (B, C, N)
- idx: (B, S)
"""

import torch


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Naive FPS in pure PyTorch.

    Args:
        xyz: (B, N, 3)
        npoint: number of points to sample
    Returns:
        indices: (B, npoint) long
    """
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"xyz must be (B, N, 3), got {tuple(xyz.shape)}")

    device = xyz.device
    B, N, _ = xyz.shape
    npoint = int(min(max(npoint, 1), N))

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather features by indices.

    Args:
        features: (B, C, N)
        idx: (B, S)
    Returns:
        gathered: (B, C, S)
    """
    if features.dim() != 3:
        raise ValueError(f"features must be (B, C, N), got {tuple(features.shape)}")
    if idx.dim() != 2:
        raise ValueError(f"idx must be (B, S), got {tuple(idx.shape)}")

    B, C, N = features.shape
    if idx.size(0) != B:
        raise ValueError("Batch size mismatch between features and idx")

    idx = idx.long().clamp(min=0, max=N - 1)
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(features, dim=2, index=idx_expanded)

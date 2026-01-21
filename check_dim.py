import torch

p = "/home/irsl/ダウンロード/test.pth"
p = "/home/irsl/URDF-Anything_CODE/checkpoints/recon/base.pth"
#p="/home/irsl/URDF-Anything_CODE/checkpoints/recon/large.pth"
p = "/home/irsl/ダウンロード/large.pth"
ckpt=torch.load(p, map_location="cpu", weights_only=True)
sd=ckpt["base_model"]  # ←ここが本体

print("num_params", len(sd))
pri = ["cls_token","pos_embed","proj","head","fc","embed","encoder","blocks","norm"]
keys = [k for k in sd.keys() if any(t in k.lower() for t in pri)]
print("sample keys:", keys[:40])

for k in keys[:120]:
    v = sd[k]
    if hasattr(v, "shape"):
        print(k, tuple(v.shape))

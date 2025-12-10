import torch
from vig_pytorch.vig import vig_ti

# 1) Instantiate a model
model = vig_ti()          # or vig_s(), vig_b(), etc.
model.eval()

# 2) Create a dummy ImageNet-sized batch
x = torch.randn(2, 3, 224, 224)   # (B=2, C=3, H=224, W=224)

# 3) Forward pass and inspect output
with torch.no_grad():
    out = model(x)

print("Type of out:", type(out))

if isinstance(out, torch.Tensor):
    print("Output shape:", out.shape)
elif isinstance(out, (list, tuple)):
    for i, o in enumerate(out):
        print(f"Out[{i}] -> type={type(o)}, shape={getattr(o, 'shape', None)}")
elif isinstance(out, dict):
    for k, v in out.items():
        print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', None)}")

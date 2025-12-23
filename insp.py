import torch
import ultralytics.nn.tasks

# Allowlist ONLY what is needed for metadata parsing
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel
])

ckpt = torch.load(
    "plate_best.pt",
    map_location="cpu",
    weights_only=True
)

print("\n=== CHECKPOINT KEYS ===")
print(ckpt.keys())

print("\n=== METADATA ===")
for k in ["ultralytics_version", "date", "task", "model", "yaml", "args"]:
    if k in ckpt:
        print(f"{k}: {ckpt[k]}")

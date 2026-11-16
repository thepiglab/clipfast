# ViT-B/16 at batch 16 / 128 / 512 (197 tokens per image) and BERT at 64 x 64 tokens.
V = [("qkv", 768, 2304, 0), ("o", 768, 768, 2), ("fc1", 768, 3072, 1), ("fc2", 3072, 768, 2)]
SCORED = [{"name": f"{n}_m{M}", "weight": 1, "args": {"M": M, "K": K, "N": N, "mode": mo}}
          for M in (3152, 25216, 100864) for n, K, N, mo in V]
NEARBY = [
    {"name": "fc1_m4096", "args": {"M": 4096, "K": 768, "N": 3072, "mode": 1}},
    {"name": "fc2_m197", "args": {"M": 197, "K": 3072, "N": 768, "mode": 2}},
    {"name": "o_m1000", "args": {"M": 1000, "K": 768, "N": 768, "mode": 2}},
    {"name": "qkv_m333", "args": {"M": 333, "K": 768, "N": 2304, "mode": 0}},
]

# BERT-base / ViT-B linears: qkv 768->2304, out 768->768, fc1 768->3072, fc2 3072->768.
DIMS = [("qkv", 768, 2304), ("o", 768, 768), ("fc1", 768, 3072), ("fc2", 3072, 768)]
SCORED = [{"name": f"{n}_m{M}", "weight": 1, "args": {"M": M, "K": K, "N": N}}
          for M in (1, 8, 16, 32, 64) for n, K, N in DIMS]
NEARBY = [
    {"name": "qkv_m3", "args": {"M": 3, "K": 768, "N": 2304}},
    {"name": "fc1_m17", "args": {"M": 17, "K": 768, "N": 3072}},
    {"name": "fc2_m40", "args": {"M": 40, "K": 3072, "N": 768}},
    {"name": "o_m100", "args": {"M": 100, "K": 768, "N": 768}},
    {"name": "proj_m5", "args": {"M": 5, "K": 640, "N": 512}},
]

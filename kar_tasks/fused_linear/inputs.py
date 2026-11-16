import torch


def make(seed, M, K, N, mode, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M, K, generator=g, device=device).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g, device=device) / K ** 0.5).to(torch.bfloat16)
    b = (0.1 * torch.randn(N, generator=g, device=device)).to(torch.bfloat16)
    r = torch.randn(M, N, generator=g, device=device).to(torch.bfloat16) if mode == 2 else \
        torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
    return x, w, b, r, mode

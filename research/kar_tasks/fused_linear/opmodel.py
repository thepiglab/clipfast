def min_bytes(M, K, N, mode):
    rd = 2 * (M * K + N * K + N) + (2 * M * N if mode == 2 else 0)
    return {"read": rd, "write": 2 * M * N}


def flops(M, K, N, mode):
    return {"tc_bf16": 2 * M * K * N}

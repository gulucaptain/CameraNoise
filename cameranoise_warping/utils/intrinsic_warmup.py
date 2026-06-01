def warmup_intrinsics_linear(K_seq, warmup=5):
    F = K_seq.shape[0]
    K_out = K_seq.clone()

    for f in range(min(warmup, F)):
        alpha = f / warmup   # 从 0 → 1
        K_out[f] = (1 - alpha) * K_seq[0] + alpha * K_seq[warmup]
    return K_out

def warmup_intrinsics_fix(K_seq, warmup=5):
    """
    K_seq: [F, 3, 3]
    warmup: 前 warmup 帧做warm-up
    """
    K_seq = K_seq.clone()
    # 取第 warmup 帧的参数作为稳定值
    K_ref = K_seq[warmup].clone()
    K_seq[:warmup] = K_ref[None]
    return K_seq

"""Block 0 check: verify PyTorch + CUDA on this machine (written for a GTX 1050 Ti / Pascal)."""
import sys
import time

import torch


def timed_matmul(dtype, n=2048, reps=20):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):  # warm-up
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000  # ms


def main() -> int:
    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA (build)  : {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available. Reinstall torch from the cu126 index (see SETUP.md).")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    print(f"GPU           : {name} ({arch}, {props.total_memory / 2**30:.1f} GiB)")
    print(f"Wheel arches  : {arch_list}")
    print(f"cuDNN         : {torch.backends.cudnn.version()} (enabled={torch.backends.cudnn.enabled})")

    if arch not in arch_list:
        print(f"WARNING: this PyTorch build has no kernels for {arch}. Use the cu126 wheels.")

    # Tensor op on GPU, checked against CPU
    x = torch.randn(512, 512)
    gpu = (x.cuda() @ x.cuda()).cpu()
    cpu = x @ x
    err = (gpu - cpu).abs().max().item()
    print(f"matmul max abs diff GPU vs CPU: {err:.2e}")
    assert err < 1e-2, "GPU result disagrees with CPU"

    # Conv + backward on GPU (exercises cuDNN)
    conv = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
    out = conv(torch.randn(4, 3, 64, 64, device="cuda"))
    out.sum().backward()
    print(f"conv fwd/bwd OK, output shape {tuple(out.shape)}")

    # FP32 vs FP16: informs the mixed_precision default for this GPU
    t32 = timed_matmul(torch.float32)
    t16 = timed_matmul(torch.float16)
    print(f"2048x2048 matmul: fp32 {t32:.2f} ms | fp16 {t16:.2f} ms")
    if t16 > t32:
        print("Note: fp16 is slower than fp32 here -> keep training.mixed_precision.enabled: false")

    print("PyTorch CUDA environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

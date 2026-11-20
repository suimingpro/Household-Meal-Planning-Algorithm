"""PyTorch设备解析与CUDA可用性检查，独立于三餐规划及遗传算法。"""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None


def resolve_device(name: str) -> torch.device:
    """解析并检查支持float64的CPU或CUDA设备。

    Args:
        name: cpu、cuda、cuda:N或auto；auto优先选可用CUDA，否则选CPU。

    Returns:
        可供张量创建使用的torch.device对象。

    Raises:
        ValueError: 设备名称无效、不受支持、CUDA不可用或GPU编号越界。
    """
    if torch is None:
        raise ValueError("PyTorch is not installed; use the NumPy CPU backend")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid PyTorch device: {name}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "Supported devices: cpu, cuda, cuda:N, auto (float64 required)"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is not available; use --device cpu or install a CUDA-enabled PyTorch build"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {device.index}")
    return device

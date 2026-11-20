"""PyTorch device resolution and CUDA availability checks, independent of meal planning and the genetic algorithm."""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None


def resolve_device(name: str) -> torch.device:
    """Resolve and check a CPU or CUDA device that supports float64.

    Args:
        name: cpu, cuda, cuda:N, or auto; auto prefers available CUDA, otherwise CPU.

    Returns:
        A torch.device object that can be used to create tensors.

    Raises:
        ValueError: Invalid or unsupported device name, unavailable CUDA, or an
            out-of-range GPU index.
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

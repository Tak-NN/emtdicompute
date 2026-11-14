"""Resolve the optional PyTorch device without requiring CUDA."""
import torch


def resolve_device(device=None):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selected = torch.device(device)
    if selected.type not in ('cpu', 'cuda'):
        raise ValueError("device must be 'cpu' or a CUDA device such as 'cuda:0'.")
    if selected.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use device='cpu' or install CUDA-enabled PyTorch on a compatible system.")
    return selected

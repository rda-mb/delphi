"""Pytorch related utilities"""
import torch


def determine_device() -> str:
    """Determines whether to use CPU or GPU. Sets up appropriate settings if GPU available."""

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return "cuda:0"
    else:
        return "cpu"

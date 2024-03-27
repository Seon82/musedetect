import warnings

import torch


def autodetect_device(warn_cpu=True) -> torch.device:
    """
    Detect the device for torch, using hardware acceleration if available.

    :param warn_cpu: Whether to warn the user if no hardware acceleration was detected.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if warn_cpu:
        warnings.warn("No hardware acceleration detected, training will be extremely slow")
    return torch.device("cpu")

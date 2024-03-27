from typing import Iterable

import torch
from torch.utils.data import Dataset, Subset


def train_test_split(dataset: Dataset, proportions: Iterable[float], seed: int | None = None) -> list[Subset]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    :param dataste: The dataset to split.
    :param proportions: Proportion of the data in each spli. Must sum to 1.
    :param seed: Optional seed for reproducibility.
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    return torch.utils.data.random_split(
        dataset,
        proportions,
        generator=generator,
    )

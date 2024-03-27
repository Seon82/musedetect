from typing import Iterable

import torch
from torchmetrics import Metric
from tqdm.auto import tqdm


def compute_metrics(
    model, loader, device, metrics: Metric | Iterable[Metric], groups_idx, instruments_idx, show_progress=False
):
    if isinstance(metrics, Metric):
        metrics = [metrics]
    model.to(device)
    metrics = {
        "flat": [metric.clone().to(device) for metric in metrics["flat"]],
        "groups": [metric.clone().to(device) for metric in metrics["groups"]],
        "instruments": [metric.clone().to(device) for metric in metrics["instruments"]],
    }

    for level in metrics:
        for metric in metrics[level]:
            metric.reset()
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, disable=not show_progress):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            for level in metrics:
                for metric in metrics[level]:
                    if level == "flat":
                        metric(pred, batch_y)
                    if level == "groups":
                        metric(pred[:, groups_idx], batch_y[:, groups_idx])
                    if level == "instruments":
                        metric(pred[:, instruments_idx], batch_y[:, instruments_idx])

    res = {}
    for level in metrics:
        res[level] = []
        for metric in metrics[level]:
            res[level].append(metric.compute())
            metric.reset()
    return res

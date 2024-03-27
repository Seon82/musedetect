import re
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch import nn
from torch.jit import TracingCheckError
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@contextmanager
def suppress_stdout(console: Console, enable: bool = True):
    if enable:
        with console.capture():
            yield
    else:
        yield


def train(
    epochs: int,
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: Optimizer,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    val_metrics_freq: int = 1,
    metrics=None,
    log_dir: Path | str = None,
    on_val_end: Callable | None = None,
    quiet: bool = False,
):
    # pylint: disable=too-many-branches,too-many-statements
    """
    Run a model's training.

    :param epochs: Number of epochs in the training loop.
    :param model: Model to train.
    :param loss_fn: Loss function to optimize.
    :param optimize: Optimizer used for the gradient descent.
    :param device: Hardware device used for training.
    :param train_loader: Training data.
    :param val_loader: Optional validation data.
    :param val_metrics_freq: Log validation metrics every val_metrics_freq epochs.
    If set to 0, nothing will be logged.
    :param show_progress: Whether to show a progress bar.
    :param log_dir: Directory for tensorboard logs.
    :param on_val_end: A function of signature executed after each
    validation loop (model, val_loss, [metric_values]) -> None.
    :param quiet: Disable all printing.
    """

    if log_dir is not None:
        log_dir = Path(log_dir)
        summary_writer = SummaryWriter(log_dir / time.strftime("%d-%m-%Y_%Hh%Mm%Ss"))

    if metrics is None:
        metrics = []

    metrics = [metric.to(device) for metric in metrics]
    global_step = 0
    console = Console()
    with suppress_stdout(console, quiet):
        for epoch in range(epochs):
            # initialize tracker variables and set our model to trainable
            console.print(f"Epoch {epoch + 1}")
            train_loss = 0
            samples = 0
            model.train()

            progress = Progress(
                TextColumn("[green]{task.completed}/{task.total}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TextColumn("{task.fields[metrics]}"),
                disable=quiet,
            )
            with progress:
                train_task_id = progress.add_task("Training", total=len(train_loader.dataset), metrics="")
                for batch_x, batch_y in train_loader:
                    (batch_x, batch_y) = (batch_x.to(device), batch_y.to(device))
                    predictions = model(batch_x)
                    if isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss)):
                        batch_y = batch_y.long()
                    else:
                        batch_y = batch_y.float()
                    loss = loss_fn(predictions, batch_y)
                    # zero the gradients accumulated from the previous steps,
                    # perform backpropagation, and update model parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # update training loss and the number of samples visited
                    train_loss += loss.item() * batch_y.size(0)
                    samples += batch_y.size(0)

                    global_step += 1
                    # Log model progress on the current training batch
                    if log_dir is not None:
                        summary_writer.add_scalar("Training loss step", loss.item(), global_step)

                    # update progress bar
                    metrics_text = f"loss: {train_loss/samples:.5f}"
                    progress.update(train_task_id, advance=len(batch_x), metrics=metrics_text, refresh=True)

            # Log model progress on the current training epoch
            if log_dir is not None:
                summary_writer.add_scalar("Training loss epoch", train_loss / samples, epoch)
                if epoch == 0:
                    try:
                        summary_writer.add_graph(model, batch_x)  # pylint: disable = undefined-loop-variable
                    except TracingCheckError:
                        warnings.warn("Adding network graph to tensorbard failed.")
                        pass

            # Compte validation metrics
            if val_loader is not None and val_metrics_freq > 0 and epoch % val_metrics_freq == 0:
                with progress:
                    val_task_id = progress.add_task("Validation", total=len(val_loader.dataset), metrics="")
                    val_loss = 0
                    samples = 0
                    model.eval()
                    for batch_x, batch_y in val_loader:
                        (batch_x, batch_y) = (batch_x.to(device), batch_y.to(device))
                        predictions = model(batch_x)
                        if isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss)):
                            batch_y = batch_y.long()
                        else:
                            batch_y = batch_y.float()
                        loss = loss_fn(predictions, batch_y)
                        val_loss += loss.item() * batch_y.size(0)
                        samples += batch_y.size(0)
                        # Update metrics
                        for metric in metrics:
                            metric(predictions, batch_y)
                        # Update progress bar
                        progress.update(val_task_id, advance=len(batch_x), metrics="", refresh=True)
                    progress.remove_task(val_task_id)

                # Display metrics
                if log_dir is not None:
                    summary_writer.add_scalar("Validation loss", val_loss / samples, epoch)
                metrics_text = f"val loss: {val_loss/samples:.5f}"
                metric_values = []
                for metric in metrics:
                    if log_dir is not None:
                        summary_writer.add_scalar(f"Validation {metric.__class__.__name__}", metric.compute(), epoch)
                    metric_name = re.sub(r"(?<!^)(?=[A-Z])", " ", metric.__class__.__name__).lower()
                    metric_values.append(metric.compute())
                    metrics_text += f" - val {metric_name}: {metric_values[-1]:.3f}"
                    metric.reset()
                console.print(metrics_text)

                # Invoke callback
                if on_val_end is not None:
                    on_val_end(model, val_loss / samples, metric_values)

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Iterable

import torch
import torchaudio.transforms as T


class ResampleDownmix:
    """
    A class used to resample audio signals and downmix them to mono.
    """

    def __init__(self, origin_sample_rate: int, new_sample_rate: int):
        """
        :param origin_sample_rate: The original signal's sample rate.
        :param new_sample_rate: The desired sample rate to resample to.
        """
        self.resampler = T.Resample(origin_sample_rate, new_sample_rate)

    def __call__(self, waveform: torch.Tensor):
        return self.resampler(waveform.mean(axis=0, keepdim=True)[0])


class Transform(ABC):
    """
    An arbitrary transformation to be applied to a tensor.
    """

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> Iterable[tuple[int, int, torch.Tensor]]:
        """
        Apply the transform to a tensor.

        Returns:
            An iterable of tuples of the form (idx_start, idx_end, transformed_tensor)
        """
        pass


class SpectralTransform(Transform, ABC):
    """
    A class of ransforms meant to cut the signal in frames and generate spectrograms for each frame.
    """

    def __init__(
        self, origin_sample_rate: int, new_sample_rate: int, window_size: int | timedelta, stride: int | timedelta
    ):
        """
        :param origin_sample_rate: The original signal's sample rate.
        :param new_sample_rate: The desired sample rate to resample to.
        :param window_size: The size of the window for frame generation. Can be passed as a number of elemnts or a time.
        :param stride: The size of the stride for frame generation. Can be passed as a number of elemnts or a time.
        """
        if isinstance(window_size, timedelta):
            window_size = int(window_size.total_seconds() * new_sample_rate)
        if isinstance(stride, timedelta):
            stride = int(stride.total_seconds() * new_sample_rate)
        self.resample_downmix = ResampleDownmix(origin_sample_rate, new_sample_rate)
        self.window_size = window_size
        self.stride = stride
        self.old_over_new = origin_sample_rate / new_sample_rate

    @abstractmethod
    def _generate_spectrum(self, waveform):
        """
        Generate one spectrogram from a waveform.
        """
        pass

    def __call__(self, waveform):
        """
        Return an iterable of (start, end, data) with start and end the index
        of samples used, and data the transformed spectrum.

        """
        waveform = self.resample_downmix(waveform)
        if waveform.shape[0] < self.window_size:
            waveform = torch.nn.functional.pad(waveform, (0, self.window_size - waveform.shape[0]), value=0)
        windowed_waveform = waveform.unfold(0, self.window_size, self.stride)
        windows_start_end = (
            (start, start + self.window_size)
            for start in range(0, waveform.size(0) - self.window_size + 1, self.stride)
        )
        all_spectrums = self._generate_spectrum(windowed_waveform)
        return (
            (int(start * self.old_over_new), int(end * self.old_over_new), mfcc)
            for mfcc, (start, end) in zip(all_spectrums, windows_start_end)
        )


class MFCCTransform(SpectralTransform):
    """
    A transform used to split the input into frames and generate an MFCC for each frame.
    """

    def __init__(self, origin_sample_rate, new_sample_rate, window_size, stride, n_mfcc, melkwargs):
        super().__init__(origin_sample_rate, new_sample_rate, window_size, stride)
        self.MFCC = T.MFCC(new_sample_rate, n_mfcc, melkwargs=melkwargs)  # pylint: disable=invalid-name

    def _generate_spectrum(self, waveform):
        return self.MFCC(waveform)


class IdentityTransform(Transform):
    """
    A minimal trasnform that returns the input tensor as-is.
    Can be used for debugging.
    """

    def __call__(self, waveform):
        return ((0, waveform.shape[1], waveform),)

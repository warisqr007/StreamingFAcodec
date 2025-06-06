import typing as tp
from dataclasses import dataclass


import torch
import math
import numpy as np
import torch.nn as nn
from torch import Tensor
import torchaudio.functional as F

from .streaming import StreamingModule
from .conv import get_extra_padding_for_conv1d, pad1d



class LinearSpectrogram(nn.Module):
    """
    Computes a POWER spectrogram (n_fft, win_length, hop_length)
    exactly as torchaudio.transforms.Spectrogram(power=2.0, normalized=False).
    """

    def __init__(self, n_fft: int, win_length: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # Hann window (no extra normalization) to match torchaudio’s default
        window = torch.hann_window(win_length, periodic=True)
        self.register_buffer("window", window, persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (batch, time)
        
        Returns:
            spec: (batch, n_fft//2 + 1, n_frames)  -- power spectrogram
        """
        B, total_length = y.shape
        device = y.device

        # How many full frames can we make?
        n_frames = (total_length - self.win_length) // self.hop_length + 1
        if n_frames <= 0:
            # Not enough samples for even one frame → return empty
            return torch.empty(B, self.n_fft // 2 + 1, 0, device=device)

        # Only keep exactly the samples needed for those n_frames
        used_length = (n_frames - 1) * self.hop_length + self.win_length
        y = y[:, :used_length]

        # Unfold into overlapping frames → shape: (B, n_frames, win_length)
        frames = y.unfold(dimension=1, size=self.win_length, step=self.hop_length)

        # Apply Hann window (no additional scaling)
        frames = frames * self.window

        # Perform real FFT along each frame, zero-padding/truncation to n_fft
        # → shape of `stft`: (B, n_frames, n_fft//2 + 1), complex dtype
        stft = torch.fft.rfft(frames, n=self.n_fft)

        # Power spectrogram: |Re + i Im|^2
        spec = stft.real.pow(2) + stft.imag.pow(2) + 1e-12  # small eps for stability

        # Reorder to (B, freq_bins, time_frames)
        spec = spec.transpose(1, 2)
        return spec


class MelSpectrogram(nn.Module):
    """
    Matches torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
        f_min=0.0,
        f_max=12000.0,   # defaults to sample_rate // 2 = 12000
        norm='slaney',
        mel_scale='htk'
    )
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        win_length: int = 1200,
        hop_length: int = 300,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        # 1) Linear power spectrogram (exact parameters)
        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length)

        # 2) Build mel‐filterbank exactly as torchaudio does:
        #    - mel_scale="htk"
        #    - norm="slaney"
        fb = F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm="slaney",
            mel_scale="htk",
        )
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, time) or (batch, time).  We’ll accept (batch, 1, T) for convenience.
        
        Returns:
            mel_spec: (batch, n_mels, n_frames), exactly the same as torchaudio’s MelSpectrogram
        """
        # If input is (B, 1, T), squeeze to (B, T)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Compute linear power spectrogram: (B, freq_bins, n_frames)
        linear_spec = self.spectrogram(x)

        if linear_spec.size(-1) == 0:
            # no frames -> return an empty mel tensor
            return torch.empty(
                x.shape[0], self.n_mels, 0, device=x.device, dtype=x.dtype
            )

        # Apply mel filterbank:
        #   1. transpose -> (B, n_frames, freq_bins)
        #   2. matmul with fb: (B, n_frames, n_freqs) @ (n_freqs, n_mels) -> (B, n_frames, n_mels)
        #   3. transpose back -> (B, n_mels, n_frames)
        mel = torch.matmul(linear_spec.transpose(-1, -2), self.fb).transpose(-1, -2)

        return mel


@dataclass
class _StreamingSpecState:
    previous: torch.Tensor | None = None

    def reset(self):
        self.previous = None


class RawStreamingMelSpectrogram(MelSpectrogram, StreamingModule[_StreamingSpecState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self.hop_length <= self.win_length
        ), "stride must be less than kernel_size."

    def _init_streaming_state(self, batch_size: int) -> _StreamingSpecState:
        return _StreamingSpecState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stride = self.hop_length
        kernel = self.win_length
        if self._streaming_state is None:
            return super().forward(input)
        else:
            # Due to the potential overlap, we might have some cache of the previous time steps.
            previous = self._streaming_state.previous
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            B, C, T = input.shape
            # We now compute the number of full convolution frames, i.e. the frames
            # that are ready to be computed.
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            offset = num_frames * stride
            # We will compute `num_frames` outputs, and we are advancing by `stride`
            # for each of the frame, so we know the data before `stride * num_frames`
            # will never be used again.
            self._streaming_state.previous = input[..., offset:]
            if num_frames > 0:
                input_length = (num_frames - 1) * stride + kernel
                out = super().forward(input[..., :input_length])
            else:
                # Not enough data as this point to output some new frames.
                out = torch.empty(
                    B, self.n_mels, 0, device=input.device, dtype=input.dtype
                )
            return out



@dataclass
class _StreamingMelSpecState:
    padding_to_add: int
    original_padding_to_add: int

    def reset(self):
        self.padding_to_add = self.original_padding_to_add


class StreamingMelSpectrogram(StreamingModule[_StreamingMelSpecState]):
    """MelSpectrogram with some builtin handling of asymmetric or causal padding
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = None,
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.conv = RawStreamingMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    @property
    def _stride(self) -> int:
        return self.conv.hop_length

    @property
    def _kernel_size(self) -> int:
        return self.conv.win_length

    @property
    def _effective_kernel_size(self) -> int:
        return self._kernel_size

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingMelSpecState:
        assert self.causal, "streaming is only supported for causal convs"
        return _StreamingMelSpecState(self._padding_total, self._padding_total)

    def forward(self, x):
        if x.dim() == 2:
            # If input is (B, T), add a channel dimension for consistency
            x = x.unsqueeze(1)
        B, C, T = x.shape
        padding_total = self._padding_total
        extra_padding = get_extra_padding_for_conv1d(
            x, self._effective_kernel_size, self._stride, padding_total
        )
        state = self._streaming_state
        if state is None:
            if self.causal:
                # Left padding for causal
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(
                    x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
                )
        else:
            if state.padding_to_add > 0 and x.shape[-1] > 0:
                x = pad1d(x, (state.padding_to_add, 0), mode=self.pad_mode)
                state.padding_to_add = 0
        return self.conv(x)

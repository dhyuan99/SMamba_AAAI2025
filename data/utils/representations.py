from abc import ABC, abstractmethod
from typing import Optional, Tuple

import math
import numpy as np
import torch as th


class RepresentationBase(ABC):
    @abstractmethod
    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def get_shape(self) -> Tuple[int, int, int]:
        ...

    @staticmethod
    @abstractmethod
    def get_numpy_dtype() -> np.dtype:
        ...

    @staticmethod
    @abstractmethod
    def get_torch_dtype() -> th.dtype:
        ...

    @property
    def dtype(self) -> th.dtype:
        return self.get_torch_dtype()

    @staticmethod
    def _is_int_tensor(tensor: th.Tensor) -> bool:
        return not th.is_floating_point(tensor) and not th.is_complex(tensor)


class StackedHistogram(RepresentationBase):
    def __init__(self, bins: int, height: int, width: int, count_cutoff: Optional[int] = None, fastmode: bool = True):
        """
        In case of fastmode == True: use uint8 to construct the representation, but could lead to overflow.
        In case of fastmode == False: use int16 to construct the representation, and convert to uint8 after clipping.

        Note: Overflow should not be a big problem because it happens only for hot pixels. In case of overflow,
        the value will just start accumulating from 0 again.
        """
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = count_cutoff
        if self.count_cutoff is None:
            self.count_cutoff = 255
        else:
            assert count_cutoff >= 1
            self.count_cutoff = min(count_cutoff, 255)
        self.fastmode = fastmode
        self.channels = 2

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('uint8')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.uint8

    def merge_channel_and_bins(self, representation: th.Tensor):
        assert representation.dim() == 4
        return th.reshape(representation, (-1, self.height, self.width))

    def get_shape(self) -> Tuple[int, int, int]:
        return 2 * self.bins, self.height, self.width

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        device = x.device
        assert y.device == pol.device == time.device == device
        assert self._is_int_tensor(x)
        assert self._is_int_tensor(y)
        assert self._is_int_tensor(pol)
        assert self._is_int_tensor(time)

        dtype = th.uint8 if self.fastmode else th.int16

        representation = th.zeros((self.channels, self.bins, self.height, self.width),
                                  dtype=dtype, device=device, requires_grad=False)

        if x.numel() == 0:
            assert y.numel() == 0
            assert pol.numel() == 0
            assert time.numel() == 0
            return self.merge_channel_and_bins(representation.to(th.uint8))
        assert x.numel() == y.numel() == pol.numel() == time.numel()

        assert pol.min() >= 0
        assert pol.max() <= 1

        bn, ch, ht, wd = self.bins, self.channels, self.height, self.width

        # NOTE: assume sorted time
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int
        t_norm = time - t0_int
        t_norm = t_norm / max((t1_int - t0_int), 1)
        t_norm = t_norm * bn
        t_idx = t_norm.floor()
        t_idx = th.clamp(t_idx, max=bn - 1)

        indices = x.long() + \
                  wd * y.long() + \
                  ht * wd * t_idx.long() + \
                  bn * ht * wd * pol.long()
        values = th.ones_like(indices, dtype=dtype, device=device)
        representation.put_(indices, values, accumulate=True)
        representation = th.clamp(representation, min=0, max=self.count_cutoff)
        if not self.fastmode:
            representation = representation.to(th.uint8)

        return self.merge_channel_and_bins(representation)


def cumsum_channel(x: th.Tensor, num_channels: int):
    for i in reversed(range(num_channels)):
        x[i] = th.sum(input=x[:i + 1], dim=0)
    return x


class MixedDensityEventStack(RepresentationBase):
    def __init__(self, bins: int, height: int, width: int, count_cutoff: Optional[int] = None,
                 allow_compilation: bool = False):
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = count_cutoff
        if self.count_cutoff is not None:
            assert isinstance(count_cutoff, int)
            assert 0 <= self.count_cutoff <= 2 ** 7 - 1

        self.cumsum_ch_opt = cumsum_channel

        if allow_compilation:
            # Will most likely not work with multiprocessing.
            try:
                self.cumsum_ch_opt = th.compile(cumsum_channel)
            except AttributeError:
                ...

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('int8')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.int8

    def get_shape(self) -> Tuple[int, int, int]:
        return self.bins, self.height, self.width

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        device = x.device
        assert y.device == pol.device == time.device == device
        assert self._is_int_tensor(x)
        assert self._is_int_tensor(y)
        assert self._is_int_tensor(pol)
        assert self._is_int_tensor(time)

        dtype = th.int8

        representation = th.zeros((self.bins, self.height, self.width), dtype=dtype, device=device, requires_grad=False)

        if x.numel() == 0:
            assert y.numel() == 0
            assert pol.numel() == 0
            assert time.numel() == 0
            return representation
        assert x.numel() == y.numel() == pol.numel() == time.numel()

        assert pol.min() >= 0  # maybe remove because too costly
        assert pol.max() <= 1  # maybe remove because too costly
        pol = pol * 2 - 1

        bn, ht, wd = self.bins, self.height, self.width

        # NOTE: assume sorted time
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int
        t_norm = (time - t0_int) / max((t1_int - t0_int), 1)
        t_norm = th.clamp(t_norm, min=1e-6, max=1 - 1e-6)
        # Let N be the number of bins. I.e. bin \in [0, N):
        # Let f(bin) = t_norm, model the relationship between bin and normalized time \in [0, 1]
        # f(bin=N) = 1
        # f(bin=N-1) = 1/2
        # f(bin=N-2) = 1/2*1/2
        # -> f(bin=N-i) = (1/2)^i
        # Also: f(bin) = t_norm
        #
        # Hence, (1/2)^(N-bin) = t_norm
        # And, bin = N - log(t_norm, base=1/2) = N - log(t_norm)/log(1/2)
        bin_float = self.bins - th.log(t_norm) / math.log(1 / 2)
        # Can go below 0 for t_norm close to 0 -> clamp to 0
        bin_float = th.clamp(bin_float, min=0)
        t_idx = bin_float.floor()

        indices = x.long() + \
                  wd * y.long() + \
                  ht * wd * t_idx.long()
        values = th.asarray(pol, dtype=dtype, device=device)
        representation.put_(indices, values, accumulate=True)
        representation = self.cumsum_ch_opt(representation, num_channels=self.bins)
        if self.count_cutoff is not None:
            representation = th.clamp(representation, min=-self.count_cutoff, max=self.count_cutoff)
        return representation


class RFFRepresentation(RepresentationBase):
    """Per-pixel RFF mean embedding of the temporal event distribution.

    Each pixel stores the mean embedding of its event timestamps in a
    D-dimensional complex space (stored as 2D real channels). This encodes
    a continuous temporal density estimate via Random Fourier Features.

    Output shape:
        two_channel  → (4*D, H, W)  — separate pos/neg embeddings
        weighted     → (2*D, H, W)  — polarity-weighted single embedding
        ignore       → (2*D, H, W)  — polarity ignored
    """

    def __init__(self, dim: int, height: int, width: int, sigma: float = 1.0,
                 polarity: str = 'two_channel', seed: int = 42):
        assert dim >= 1
        assert polarity in ('two_channel', 'weighted', 'ignore')
        self.dim = dim
        self.height = height
        self.width = width
        self.sigma = sigma
        self.polarity = polarity

        rng = np.random.default_rng(seed)
        T_np = rng.normal(0.0, sigma, size=dim).astype(np.float32)
        self._T = th.tensor(T_np)  # (D,) on CPU; moved to device on first use

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('float32')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.float32

    def get_shape(self) -> Tuple[int, int, int]:
        C = 4 * self.dim if self.polarity == 'two_channel' else 2 * self.dim
        return C, self.height, self.width

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        device = x.device
        D = self.dim
        C = 4 * D if self.polarity == 'two_channel' else 2 * D
        num_pixels = self.height * self.width

        if x.numel() == 0:
            return th.zeros(C, self.height, self.width, dtype=th.float32, device=device)

        T = self._T.to(device)  # (D,)

        t = time.float()
        t_min, t_max = t.min(), t.max()
        delta = (t_max - t_min).clamp(min=1.0)
        t_norm = (t - t_min) / delta  # (n,)

        phase = t_norm.unsqueeze(1) * T.unsqueeze(0)  # (n, D)
        cos_p = th.cos(phase)
        sin_p = th.sin(phase)

        pixel_idx = y.long() * self.width + x.long()  # (n,)

        if self.polarity == 'two_channel':
            result = th.zeros(4 * D, num_pixels, dtype=th.float32, device=device)
            count_pos = th.zeros(num_pixels, dtype=th.float32, device=device)
            count_neg = th.zeros(num_pixels, dtype=th.float32, device=device)

            pos_mask = pol > 0
            neg_mask = ~pos_mask

            if pos_mask.any():
                idx = pixel_idx[pos_mask]
                result[:D].scatter_add_(1, idx.unsqueeze(0).expand(D, -1), cos_p[pos_mask].T)
                result[D:2*D].scatter_add_(1, idx.unsqueeze(0).expand(D, -1), sin_p[pos_mask].T)
                count_pos.scatter_add_(0, idx, th.ones(idx.numel(), dtype=th.float32, device=device))

            if neg_mask.any():
                idx = pixel_idx[neg_mask]
                result[2*D:3*D].scatter_add_(1, idx.unsqueeze(0).expand(D, -1), cos_p[neg_mask].T)
                result[3*D:].scatter_add_(1, idx.unsqueeze(0).expand(D, -1), sin_p[neg_mask].T)
                count_neg.scatter_add_(0, idx, th.ones(idx.numel(), dtype=th.float32, device=device))

            mask_p = count_pos > 0
            result[:2*D, mask_p] /= count_pos[mask_p].unsqueeze(0)
            mask_n = count_neg > 0
            result[2*D:, mask_n] /= count_neg[mask_n].unsqueeze(0)

        else:
            result = th.zeros(2 * D, num_pixels, dtype=th.float32, device=device)
            count = th.zeros(num_pixels, dtype=th.float32, device=device)

            if self.polarity == 'weighted':
                w = pol.float() * 2.0 - 1.0  # {0,1} → {-1,+1}
                cos_w = cos_p * w.unsqueeze(1)
                sin_w = sin_p * w.unsqueeze(1)
            else:
                cos_w, sin_w = cos_p, sin_p

            idx_exp = pixel_idx.unsqueeze(0).expand(D, -1)
            result[:D].scatter_add_(1, idx_exp, cos_w.T)
            result[D:].scatter_add_(1, idx_exp, sin_w.T)
            count.scatter_add_(0, pixel_idx, th.ones(x.numel(), dtype=th.float32, device=device))

            mask = count > 0
            result[:, mask] /= count[mask].unsqueeze(0)

        return result.reshape(C, self.height, self.width)

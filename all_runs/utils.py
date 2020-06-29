import torch

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve


def to_numpy(torch_tensor):
    return torch_tensor.detach().numpy()


def to_torch(numpy_array):
    return torch.from_numpy(numpy_array).unsqueeze(1)


class EdgeDecoder(object):

    def __init__(self, size=21, filter_scale=4., num_orients=16):
        self.filters = self.get_gabor_filters(
            size=size, filter_scale=filter_scale, num_orients=num_orients, weights=True
        )


    def decode_edge_map(self, frcs, data):
        frcs = to_numpy(frcs)

        edge_map_list = []

        for i, (cnt) in enumerate(frcs):
            data_i = data[i]
            shape = (data_i.size(1), data_i.size(2))
            edge_map = decode(cnt, self.filters, shape)
            edge_map_list.append(edge_map)

        return to_torch(np.array(edge_map_list))


    # From https://github.com/vicariousinc/science_rcn/blob/master/science_rcn/preproc.py
    def get_gabor_filters(self, size=21, filter_scale=4., num_orients=16, weights=False):

        def _get_sparse_gaussian():
            """Sparse Gaussian."""
            size = 2 * np.ceil(np.sqrt(2.) * filter_scale) + 1
            alt = np.zeros((int(size), int(size)), np.float32)
            alt[int(size // 2), int(size // 2)] = 1
            gaussian = gaussian_filter(alt, filter_scale / np.sqrt(2.), mode='constant')
            gaussian[gaussian < 0.05 * gaussian.max()] = 0
            return gaussian

        gaussian = _get_sparse_gaussian()
        filts = []
        for angle in np.linspace(0., 2 * np.pi, num_orients, endpoint=False):
            acts = np.zeros((size, size), np.float32)
            x, y = np.cos(angle) * filter_scale, np.sin(angle) * filter_scale
            acts[int(size / 2 + y), int(size / 2 + x)] = 1.
            acts[int(size / 2 - y), int(size / 2 - x)] = -1.
            filt = fftconvolve(acts, gaussian, mode='same')
            filt /= np.abs(filt).sum()  # Normalize to ensure the maximum output is 1
            if weights:
                filt = np.abs(filt)
            filts.append(filt)
        return filts


# From https://github.com/vicariousinc/science_rcn/blob/master/science_rcn/inference.py
def decode(backtrace_positions, filters, shape):

    height, width = shape
    f_h, f_w = filters[0].shape
    layers = np.zeros((len(backtrace_positions), height, width))
    fo_h, fo_w = f_h // 2, f_w // 2
    from_r, to_r = (np.maximum(0, backtrace_positions[:, 1] - fo_h),
                    np.minimum(height, backtrace_positions[:, 1] - fo_h + f_h))
    from_c, to_c = (np.maximum(0, backtrace_positions[:, 2] - fo_w),
                    np.minimum(width, backtrace_positions[:, 2] - fo_w + f_w))
    from_fr, to_fr = (np.maximum(0, fo_h - backtrace_positions[:, 1]),
                      np.minimum(f_h, height - backtrace_positions[:, 1] + fo_h))
    from_fc, to_fc = (np.maximum(0, fo_w - backtrace_positions[:, 2]),
                      np.minimum(f_w, width - backtrace_positions[:, 2] + fo_w))

    assert np.all(to_r - from_r == to_fr - from_fr)
    assert np.all(to_c - from_c == to_fc - from_fc)

    canvas = np.zeros((height, width))
    for i, (f, r, c) in enumerate(backtrace_positions):
        # Convolve sparse top-down activations with filters
        try:
            filt = filters[f][from_fr[i]:to_fr[i], from_fc[i]:to_fc[i]]
            canvas[from_r[i]:to_r[i], from_c[i]:to_c[i]] += filt

        except Exception as e:
            pass
        
    return canvas
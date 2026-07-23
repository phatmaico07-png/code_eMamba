"""Datasets for eMamba-MARS.

SyntheticMARS  -- random data with a LEARNABLE low-rank input->pose mapping,
                 used for the smoke-train (proves the pipeline learns; no real
                 data needed).
MARSDataset    -- real MARS data loader. Expects feature (N,8,8,5) and label
                 (N,57). If labels store all 25 Kinect joints (N,75), the 19
                 used joints are selected via KINECT25_TO_19.
"""
import os
import torch
from torch.utils.data import Dataset

# our 19-joint order -> original Kinect V2 (25) indices (see docs/06_mars_skeleton_tree.md)
KINECT25_TO_19 = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]


class SyntheticMARS(Dataset):
    """Synthetic data with a LEARNABLE low-rank input->pose mapping.

    A latent vector z (dim < D) drives BOTH the input (linearly) and the target.
    A pooling head can recover z from the pooled features and map it to the pose,
    so the loss drops clearly -- a meaningful smoke test for this architecture.
    (A dense random 320->57 map is NOT learnable here because the head mean-pools
    the 16 tokens into a single 20-vector.)
    """

    def __init__(self, n=2048, latent=12, scale=20.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, latent, generator=g)
        A = torch.randn(latent, 8 * 8 * 5, generator=g) / (latent ** 0.5)
        X = (z @ A).reshape(n, 8, 8, 5) + 0.05 * torch.randn(n, 8, 8, 5, generator=g)
        Bm = torch.randn(latent, 57, generator=g) / (latent ** 0.5)
        Y = (z @ Bm) * scale + 0.1 * torch.randn(n, 57, generator=g)   # ~cm scale
        self.X, self.Y = X, Y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def _load_array(path):
    import numpy as np
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        z = np.load(path)
        return z[list(z.keys())[0]]
    if ext == ".mat":
        from scipy.io import loadmat
        m = loadmat(path)
        keys = [k for k in m.keys() if not k.startswith("__")]
        return m[keys[0]]
    raise ValueError("unsupported file: " + str(path))


class MARSDataset(Dataset):
    """Real MARS data.

    Official feature/*.npy: feature (N,8,8,5); label (N,57) in METERS with
    AXIS-MAJOR layout [X(19) | Y(19) | Z(19)] (confirmed from MARS_model.py).
    We convert labels to joint-major [x0,y0,z0, x1,y1,z1, ...] (to match the
    kinematic head) and scale to cm (unit_scale=100, so MAE/RMSE print in cm
    like the paper).
    """
    def __init__(self, feature_path, label_path, select_19_from_25=True,
                 label_layout="axis_major", unit_scale=100.0):
        import numpy as np
        X = np.asarray(_load_array(feature_path)).astype("float32")
        Y = np.asarray(_load_array(label_path)).astype("float32")
        if X.ndim == 2:                       # (N, 320) -> (N,8,8,5)
            X = X.reshape(-1, 8, 8, 5)
        did_select = False
        if Y.shape[-1] == 75 and select_19_from_25:   # raw 25-joint labels
            Y = Y.reshape(Y.shape[0], 25, 3)[:, KINECT25_TO_19].reshape(Y.shape[0], 57)
            did_select = True
        if (not did_select) and label_layout == "axis_major":
            Y = Y.reshape(Y.shape[0], 3, 19).transpose(0, 2, 1).reshape(Y.shape[0], 57)
        Y = Y * float(unit_scale)
        assert Y.shape[-1] == 57, "expected 57 label dims, got %d" % Y.shape[-1]
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

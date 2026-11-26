"""Public plotting functions return figures (Agg backend) for 2, 3 and 4 dimensions and write no file."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from hydyreg import plotting  # noqa: E402
from hydyreg.trajectories import Trajectories  # noqa: E402


class _Lin:
    """Minimal 'learned system' for plot_comparison: x' = -x."""
    def simulate_trajectory(self, h, T, x0s, max_step_factor=10):
        t = np.arange(0, T + h / 2, h)
        return Trajectories([t], [np.exp(-t)[:, None] * np.asarray(x0s[0])[None, :]])


class _Model:
    def __init__(self, D):
        rng = np.random.default_rng(0)
        self.Xfit_ = rng.normal(size=(60, D))
        self.modesfit_ = (self.Xfit_[:, 0] > 0).astype(int)
        self.X_classify_ = self.Xfit_
        self.n = 2
        self.labelfun_ = lambda x: int(x[0] > 0)


@pytest.mark.parametrize("D", [2, 3, 4])
def test_public_functions(D, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = np.arange(0, 2, 0.05)
    tr = Trajectories([t, t], [np.exp(-t)[:, None] * np.ones(D), np.exp(-t)[:, None] * np.linspace(-1, 1, D)])
    figs = [plotting.plot_trajectories(tr)[0], plotting.plot_trajectories(tr, labelfun=lambda x: int(x[0] > 0))[0],
            plotting.plot_comparison(None, _Lin(), tr)[0], plotting.plot_modes(_Model(D))[0]]
    seg = Trajectories([t], [tr.x_list[0]])
    figs.append(plotting.plot_segmentation(seg, [[10]])[0])
    if D == 2:
        figs.append(plotting.plot_regions(_Model(2), trajectories=tr)[0])
    assert all(isinstance(f, plt.Figure) for f in figs)
    assert os.listdir(tmp_path) == []
    plt.close("all")

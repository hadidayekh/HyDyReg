"""Result of a learning run."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

STAGES = ("passive", "refined", "final")


@dataclass
class LearnResult:
    """What a learning run produces.

    model: the fitted HybridRegressor. system: the learned system (an SNDS that can be simulated).
    snapshots: copies of the model after the passive step ('passive') and after boundary refinement ('refined');
        'final' is `model`.
    errors: per stage, {'train': (mean, max), 'test': (mean, max)} relative differences of trajectories
        (test only when a test set exists).
    rd_dot: mean relative difference of the derivatives on the test trajectories (None without a reference).
    times: seconds of the passive step ('passive'), of boundary refinement ('refinement') and of the
        equivalence queries ('equivalence').
    n_eq: number of equivalence queries. terminated: the last query found no counterexample (None without
        queries). info: run information (noise estimate, windows, deltas, boundary statistics, ...).
    """
    model: Any
    system: Any
    config: Any
    name: str = ""
    snapshots: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rd_dot: Optional[float] = None
    times: Dict[str, float] = field(default_factory=dict)
    n_eq: int = 0
    terminated: Optional[bool] = None
    info: Dict[str, Any] = field(default_factory=dict)
    train_trajectories: Any = None
    test_trajectories: Any = None
    reference_system: Any = None
    names: Any = None

    @property
    def n_modes(self) -> int:
        return int(self.model.n)

    def equations(self, fmt: str = "text", digits: int = 4) -> str:
        """The learned dynamics and boundaries, as text (fmt='text') or LaTeX (fmt='latex', standard amsmath,
        to paste into a document; `model.gen_full_latex_eq()` gives the format of the thesis)."""
        from ..export.text import model_text, model_latex
        if fmt == "latex":
            return model_latex(self.model, names=self.names, digits=digits)
        return model_text(self.model, names=self.names, digits=digits)

    def summary(self) -> str:
        lines = [f"{self.name or 'run'}: {self.n_modes} mode(s)"]
        if self.terminated is not None:
            lines[0] += f", {self.n_eq} equivalence quer{'y' if self.n_eq == 1 else 'ies'}, " + \
                        ("terminated" if self.terminated else "NOT terminated")
        for stage in STAGES:
            e = self.errors.get(stage)
            if e:
                parts = [f"{k} RD mean {v[0]:.3g} max {v[1]:.3g}" for k, v in e.items()]
                lines.append(f"  {stage:8s} " + "; ".join(parts))
        if self.rd_dot is not None:
            lines.append(f"  derivative RD on the test set: {self.rd_dot:.3g}")
        return "\n".join(lines)

    def to_row(self) -> list:
        """The 11 values of a row of the thesis tables: d1, d2, d3 (mean), d1, d2, d3 (max), t1, t2, t3, NEQ, d_dot."""
        e = self.errors
        t = [e[s]["test"] for s in STAGES]
        return [t[0][0], t[1][0], t[2][0], t[0][1], t[1][1], t[2][1],
                self.times["passive"], self.times["refinement"], self.times["equivalence"], self.n_eq, self.rd_dot]

    def plot(self, kind="comparison", **kwargs):
        """Shortcut to hydyreg.plotting (kind: 'comparison', 'regions', 'modes')."""
        from .. import plotting
        kwargs.setdefault("names", self.names)
        if kind == "regions":
            kwargs.setdefault("trajectories", self.train_trajectories)
            return plotting.plot_regions(self.model, **kwargs)
        if kind == "modes":
            return plotting.plot_modes(self.model, **kwargs)
        trajs = self.test_trajectories if self.test_trajectories is not None else self.train_trajectories
        return plotting.plot_comparison(self.reference_system, self.system, trajs, **kwargs)

"""Command line: fit, simulate, export, bench list, configuration errors."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"


def hydyreg(*args, cwd=None):
    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, "-m", "hydyreg.cli", *map(str, args)], capture_output=True, text=True,
                          cwd=cwd or ROOT, env=env)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    out = tmp_path_factory.mktemp("fit")
    p = hydyreg("fit", DATA / "isolette_1.csv", DATA / "isolette_2.csv", "--seg-tol", "0.01", "--out", out, "--latex", "-q")
    assert p.returncode == 0, p.stderr[-2000:]
    return out


def test_fit_recovers_the_two_modes(fitted):
    model = json.loads((fitted / "model.json").read_text())
    assert model["n_modes"] == 2 and model["names"] == ["T_box", "T_heater"]
    coefs = model["coefficients"]
    for c in coefs:
        A, b = np.array(c["coef"]), np.array(c["intercept"])
        assert np.allclose(A, [[-0.026, 0.026], [0, 0]], atol=1e-3)
        assert np.isclose(abs(b[1]), 1, atol=1e-3) and abs(b[0]) < 1e-3
    assert {np.sign(c["intercept"][1]) for c in coefs} == {-1.0, 1.0}
    for f in ("report.txt", "run.json", "equations.tex"):
        assert (fitted / f).exists()


def test_simulate_and_export(fitted, tmp_path):
    p = hydyreg("simulate", fitted / "model.json", "--x0", "100,90", "--step", "0.25", "--time", "5", "--out", tmp_path / "s.csv")
    assert p.returncode == 0, p.stderr
    data = np.loadtxt(tmp_path / "s.csv", delimiter=",", skiprows=1)
    assert data.shape == (21, 3)
    for fmt in ("text", "latex"):
        p = hydyreg("export", fitted / "model.json", "--format", fmt)
        assert p.returncode == 0 and "T_" in p.stdout


def test_config_file_and_errors(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('method = "incremental"\nseg_tol = 0.01\n[noise]\nadapt = true\nwrong_key = 1\n')
    p = hydyreg("fit", DATA / "isolette_1.csv", "--config", cfg, "--out", tmp_path / "o")
    assert p.returncode == 2 and "wrong_key" in p.stderr and "adapt" in p.stderr
    p = hydyreg("fit", DATA / "isolette_1.csv", "--set", "method=foo", "--out", tmp_path / "o")
    assert p.returncode == 2 and "method" in p.stderr


def test_bench_list():
    p = hydyreg("bench", "list")
    assert p.returncode == 0 and p.stdout.count("parametric") >= 10 and "incr/isolette" in p.stdout


def test_learn_isolette(tmp_path):
    p = hydyreg("learn", "--system", "hydyreg.benchmarks.systems:isolette", "--config",
                ROOT / "examples" / "configs" / "learn_isolette.toml", "--out", tmp_path, "-q")
    assert p.returncode == 0, p.stderr[-2000:]
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["terminated"] is True and run["n_modes"] == 2

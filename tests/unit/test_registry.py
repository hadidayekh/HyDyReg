"""The row registry keeps the settings recorded in tests/golden/ref/row_settings.json (a change of a row's
configuration, design or seeding must be deliberate: re-record with `capture.py --row-settings`), and passive
learning from simulated data gives the dynamics of active learning."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
sys.path.insert(0, str(GOLDEN))
import adapter  # noqa: E402
from hashing import jsonable  # noqa: E402

REF = json.loads((GOLDEN / "ref" / "row_settings.json").read_text())


def test_every_row_is_recorded():
    from hydyreg.benchmarks.rows import ROW_IDS
    assert sorted(REF) == sorted(ROW_IDS)


@pytest.mark.parametrize("row_id", list(REF))
def test_row_settings(row_id):
    assert jsonable(adapter.row_settings(row_id)) == REF[row_id]


def test_passive_learning_gives_the_dynamics_of_active_learning():
    from hydyreg.benchmarks.rows import get_row, run_row
    from hydyreg.learning.pipeline import learn_from_data
    row = get_row("incr/isolette")
    active = run_row(row, print_on=False)
    data = row.system().simulate_trajectory(row.design.time_step, row.design.time_range, row.design.x0s)
    passive = learn_from_data(data, get_row("incr/isolette").config, print_on=False)
    a, p = active.snapshots["refined"], passive.snapshots["refined"]
    assert np.array_equal(a.Omegas_, p.Omegas_) and np.array_equal(a.betas_, p.betas_)
    assert np.array_equal(a.classifier_.coef_, p.classifier_.coef_)
    assert passive.terminated is None and passive.info["boundary"].get("unresolved", 0) == 0
    assert passive.errors["refined"]["train"][0] < 1e-4

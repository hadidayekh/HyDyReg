"""Golden cases: short learning runs whose numbers are recorded in tests/golden/ref/ (regression smoke tests).

Each case is a benchmark row with overrides: keys of LearnConfig ('equivalence.max_rounds') or of the sampling
design ('design.noise_std'). A case does not have to terminate or give a good model; it only has to give the
same numbers as its reference until a deliberate change of results. Five fast cases: the incremental and the
optimization-based method on a three-mode system, measurement noise, the clusterwise solver, the decision list.
Correctness is checked by tests/unit/test_correctness.py.
`seconds` is the measured run time of the case on the reference machine.
"""

CASES = {
    "G3": dict(row="incr/affine_3modes", overrides={}, slow=False),
    "G4": dict(row="opti/affine_3modes", overrides={}, slow=False),
    "G5": dict(row="incr/isolette", overrides={"design.noise_std": 0.01, "equivalence.max_rounds": 3}, slow=False),
    "G9": dict(row="opti/isolette", overrides={"optimization.solver": "clusterwise"}, slow=False),
    "G13": dict(row="incr/affine_3modes", overrides={"classifier.kind": "decision_list"}, slow=False),
}

# Measured run times (seconds, reference machine, without plots).
SECONDS = dict(G3=10, G4=13, G5=10, G9=10, G13=8)

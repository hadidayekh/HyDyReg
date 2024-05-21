from typing import List

class Polytope:
    def __init__(self) -> None:
        self.hyperplanes = []
        pass

class IncrementalClassifier:
    def __init__(self) -> None:
        self.regions : List[Polytope] = []
        pass
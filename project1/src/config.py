from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 8004
    n_splits: int = 5
    n_repeats: int = 1
    n_jobs: int = -1


config = Config()

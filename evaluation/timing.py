# Context manager timer and TimingTracker for recording per-run train/pred times.
import time
import pandas as pd
import os
from contextlib import contextmanager


class _Timer:
    elapsed = 0.0


@contextmanager
def timer():
    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


class TimingTracker:

    def __init__(self):
        self._rows = []

    def record(self, model, dataset, sample_size, run, train_time, pred_time):
        self._rows.append({
            "model":        model,
            "dataset":      dataset,
            "sample_size":  sample_size,
            "run":          run,
            "train_time_s": train_time,
            "pred_time_s":  pred_time,
        })

    def to_dataframe(self):
        return pd.DataFrame(self._rows)

    def save_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

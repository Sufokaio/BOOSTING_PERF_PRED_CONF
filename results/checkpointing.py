# Persists completed (dataset, model, sample_size, run) tuples to JSON for crash recovery.
import json
import os


class Checkpoint:

    def __init__(self, path):
        self.path = path
        self._done = set()
        self.load()

    def _key(self, dataset, model, sample_size, run):
        return f"{dataset}|{model}|{sample_size}|{run}"

    def is_done(self, dataset, model, sample_size, run) -> bool:
        return self._key(dataset, model, sample_size, run) in self._done

    def mark_done(self, dataset, model, sample_size, run):
        self._done.add(self._key(dataset, model, sample_size, run))

    def save(self):
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(list(self._done), f)

    def __len__(self):
        return len(self._done)

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._done = set(json.load(f))
            except (json.JSONDecodeError, IOError):
                self._done = set()

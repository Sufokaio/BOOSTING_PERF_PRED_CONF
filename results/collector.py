# Thread-safe result accumulator, appends to CSV after every completed run.
import os
import threading
import pandas as pd


_COLUMNS = ["dataset", "model", "sample_size", "run", "metric", "value",
            "train_time_s", "pred_time_s"]


class ResultCollector:

    def __init__(self):
        self._lock = threading.Lock()
        self._rows = []
        self._completed = set()

    def record(self, dataset, model, sample_size, run,
               metrics_dict, train_time, pred_time):
        key = (dataset, model, sample_size, run)
        with self._lock:
            for metric, value in metrics_dict.items():
                self._rows.append({
                    "dataset":     dataset,
                    "model":       model,
                    "sample_size": sample_size,
                    "run":         run,
                    "metric":      metric,
                    "value":       value,
                    "train_time_s": train_time,
                    "pred_time_s":  pred_time,
                })
            self._completed.add(key)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with self._lock:
            if not self._rows:
                return
            df_new = pd.DataFrame(self._rows, columns=_COLUMNS)

        if os.path.exists(path):
            try:
                df_existing = pd.read_csv(path)
                existing_keys = set(
                    zip(df_existing["dataset"], df_existing["model"],
                        df_existing["sample_size"], df_existing["run"],
                        df_existing["metric"])
                )
                df_new = df_new[
                    ~df_new.apply(
                        lambda r: (r["dataset"], r["model"],
                                   r["sample_size"], r["run"], r["metric"])
                        in existing_keys,
                        axis=1,
                    )
                ]
                if df_new.empty:
                    return
                df_new.to_csv(path, mode="a", header=False, index=False)
            except Exception:
                df_new.to_csv(path, index=False)
        else:
            df_new.to_csv(path, index=False)

    def load(self, path):
        if not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        with self._lock:
            for _, row in df.iterrows():
                self._rows.append(row.to_dict())
                self._completed.add(
                    (row["dataset"], row["model"], row["sample_size"], row["run"])
                )

    def get_completed_runs(self):
        with self._lock:
            return set(self._completed)

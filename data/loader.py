import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
   


class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._X = None
        self._y = None
        self._load()

    def _load(self):
        df = pd.read_csv(self.csv_path)
        arr = df.values.astype(float)
        self._X = arr[:, :-1]
        self._y = arr[:, -1]

    def load(self):
        return self._X.copy(), self._y.copy()

    def sample(self, size, seed):
        n = len(self._y)
        if size >= n:
            raise ValueError(
                f"Sample size {size} must be strictly less than dataset size {n} "
                f"(file: {self.csv_path})"
            )
        rng = np.random.RandomState(seed)
        indices = rng.choice(n, size=size, replace=False)
        test_indices = np.setdiff1d(np.arange(n), indices)

        X_train = self._X[indices]
        y_train = self._y[indices]
        X_test  = self._X[test_indices]
        y_test  = self._y[test_indices]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

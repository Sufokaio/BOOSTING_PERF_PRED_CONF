from sklearn.neural_network import MLPRegressor
from models.boosting_framework import GradientBoostingFramework
from config import MLP_PARAMS, DEFAULT_M, DEFAULT_ETA


class MLPBoost(GradientBoostingFramework):
    def __init__(self, M=DEFAULT_M, eta=DEFAULT_ETA):
        super().__init__(MLPRegressor, dict(MLP_PARAMS), M, eta)

    def _make_learner(self, stage=0):
        params = dict(self.base_learner_params)
        params["random_state"] = params.get("random_state", 0) + stage
        return self.base_learner_class(**params)

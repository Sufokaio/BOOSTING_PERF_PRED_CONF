from sklearn.linear_model import Ridge
from models.boosting_framework import GradientBoostingFramework
from config import RIDGE_PARAMS, DEFAULT_M, DEFAULT_ETA


class RidgeLRBoost(GradientBoostingFramework):
    def __init__(self, M=DEFAULT_M, eta=DEFAULT_ETA):
        super().__init__(Ridge, dict(RIDGE_PARAMS), M, eta)

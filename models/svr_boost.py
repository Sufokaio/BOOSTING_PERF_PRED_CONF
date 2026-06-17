from sklearn.svm import SVR
from models.boosting_framework import GradientBoostingFramework
from config import SVR_PARAMS, DEFAULT_M, DEFAULT_ETA


class SVRBoost(GradientBoostingFramework):
    def __init__(self, M=DEFAULT_M, eta=DEFAULT_ETA):
        super().__init__(SVR, dict(SVR_PARAMS), M, eta)

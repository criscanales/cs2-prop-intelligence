"""Platt scaling on logit(p): smooth and monotone, so probabilities don't collapse into isotonic steps."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y):
        self.lr = LogisticRegression(C=100.0).fit(_logit(p)[:, None], np.asarray(y))
        return self

    def predict(self, p):
        return np.clip(self.lr.predict_proba(_logit(p)[:, None])[:, 1], 0.01, 0.99)

    def to_params(self) -> dict:
        return {"coef": float(self.lr.coef_[0, 0]), "intercept": float(self.lr.intercept_[0])}

    @classmethod
    def from_params(cls, d: dict) -> "Platt":
        """Rebuild from to_params() output (plain numbers, so bundles never need unpickling)."""
        lr = LogisticRegression(C=100.0)
        lr.coef_, lr.intercept_ = np.array([[float(d["coef"])]]), np.array([float(d["intercept"])])
        lr.classes_, lr.n_features_in_ = np.array([0, 1]), 1
        obj = cls()
        obj.lr = lr
        return obj

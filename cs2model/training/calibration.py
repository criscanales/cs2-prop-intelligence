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

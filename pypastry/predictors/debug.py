import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DebugTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        print(X)
        return X

    def fit(self, X, y=None, **fit_params):
        return self

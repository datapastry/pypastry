from pypastry.predictors.debug import DebugTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline


class DocumentClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, text_columns):
        self.pipeline = Pipeline([
            ('column', ColumnTransformer([
                ('count', CountVectorizer(), text_columns),
            ])),
            # ('tfidf', TfidfTransformer()),
            ('After tfidf', DebugTransformer()),
            ('clf', SGDClassifier(tol=1e-3)),
        ])

    def fit(self, X, y):
        return self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)


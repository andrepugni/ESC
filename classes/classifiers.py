# standard modules
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
import copy
import pandas as pd


# pluginrule
class PlugInRule(ClassifierMixin, BaseEstimator):
    """
    Class for PlugIn algorithm.
    It takes as input a probabilistic classifier and it constructs the PlugIn estimator of Herbei and Wegkamp
    References

    Example
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.linear_model import LogisticRegression
    >>> from classes.classifiers import PlugInRule
    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> clf = PlugInRule(model=LogisticRegression())
    >>> clf.fit(X_tr, y_tr)
    >>> preds = clf.predict(X_te)
    """

    def __init__(
        self,
        model,
        coverages: list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75],
        seed: int = 42,
    ):
        """

        :param model:
        :param coverages:
        :param seed:
        """
        self.model = model
        self.coverages = sorted(coverages, reverse=True)
        self.seed = seed
        self.thetas = None

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        test_perc: float = 0.1,
        confidence_function: str = "softmax",
    ):
        """

        :param X: numpy.array or pandas.DataFrame
                The training features
        :param y: numpy.array of pandas.Series
                The training target variable
        :param sample_weight:
                The weights to apply to instances.
                The default is None.
        :param test_perc: float
                The percentage of instances in the available data to use as a holdout set.
                The default is .1.
        :param confidence_function: string
                The type of confidence function to use.
                Available options are:
                    - 'softmax': it takes the maximum predicted value from the .predict_proba() of the base classifier
                The default is 'softmax'.
        :return:
                The fitted classifier h(x) over 1-test_perc training set
        """

        # here we store the classes
        self.classes = np.unique(y)
        # here we split train and holdout
        X_train, X_hold, y_train, y_hold = train_test_split(
            X, y, stratify=y, random_state=self.seed, test_size=test_perc
        )
        self.model.fit(X_train, y_train)
        # quantiles
        probas = self.model.predict_proba(X_hold)
        if confidence_function == "softmax":
            self.confidence = "softmax"
            confs = np.max(probas, axis=1)
        else:
            raise NotImplementedError("Confidence function not yet implemented")
        self.quantiles = [1 - c for c in self.coverages]
        self.thetas = [np.quantile(confs, q) for q in self.quantiles]

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(
                "The original model does not have predict_proba method."
            )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def qband(self, X, confidence_function: str = "softmax"):
        if self.thetas is not None:
            if self.confidence == "softmax":
                confs = np.max(self.predict_proba(X), axis=1)
            else:
                raise AttributeError(
                    "The original model does not have predict_proba method."
                )
            return np.digitize(confs, self.thetas)
        else:
            raise ValueError(
                "The model is not fitted yet. Please call the fit method before."
            )


class SCRoss(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model,
        cv: int = 5,
        coverages: list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75],
        seed: int = 42,
    ):
        self.cv = cv
        self.seed = seed
        self.coverages = sorted(coverages, reverse=True)
        self.thetas = None
        self.kmodels = [copy.deepcopy(model) for _ in range(self.cv)]
        self.model = copy.deepcopy(model)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        confidence_function: str = "softmax",
        quantile_est: str = "knight",
        n_jobs: int = 1,
    ):
        self.classes_ = np.unique(y)
        z = []
        localthetas = []
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        if n_jobs == 1:
            for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                if isinstance(X, pd.DataFrame):
                    X_train = X.iloc[train_index]
                    X_test = X.iloc[test_index]
                else:
                    X_train = X[train_index]
                    X_test = X[test_index]
                if isinstance(y, pd.Series):
                    y_train = y.iloc[train_index]
                    # y_test = y.iloc[test_index]
                else:
                    y_train = y[train_index]
                    # y_test = y[test_index]
                self.kmodels[i].fit(X_train, y_train)
                # quantiles
                probas = self.kmodels[i].predict_proba(X_test)
                if confidence_function == "softmax":
                    self.confidence = "softmax"
                    confs = np.max(probas, axis=1)
                else:
                    raise NotImplementedError(
                        "Confidence function not yet implemented."
                    )
                z.append(confs)
        elif n_jobs == -1:
            raise NotImplementedError("Parallelization not yet implemented.")
        elif n_jobs > 1:
            raise NotImplementedError("Parallelization not yet implemented.")
        self.z = z
        confs = np.concatenate(z).ravel()
        self.quantiles = [1 - cov for cov in self.coverages]
        if quantile_est == "knight":
            sub_confs_1, sub_confs_2 = train_test_split(
                confs, test_size=0.5, random_state=42
            )
            tau = 1 / np.sqrt(2)
            self.thetas = [
                (
                    tau * np.quantile(confs, q)
                    + (1 - tau)
                    * (
                        0.5 * np.quantile(sub_confs_1, q)
                        + 0.5 * np.quantile(sub_confs_2, q)
                    )
                )
                for q in self.quantiles
            ]
        elif quantile_est == "standard":
            self.thetas = [np.quantile(confs, q) for q in self.quantiles]
        else:
            raise NotImplementedError("Quantile estimator not yet implemented")
        self.model.fit(X, y)

    def predict_proba(self, X, ensembling=False):
        if ensembling:
            if hasattr(self.kmodels[0], "predict_proba"):
                return np.mean([clf.predict_proba(X) for clf in self.kmodels], axis=0)
            else:
                raise AttributeError(
                    "The original model does not have predict_proba method."
                )
        else:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            else:
                raise AttributeError(
                    "The original model does not have predict_proba method."
                )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def qband(self, X):
        if self.thetas is not None:
            if self.confidence == "softmax":
                confs = np.max(self.predict_proba(X), axis=1)
            else:
                raise AttributeError(
                    "The original model does not have predict_proba method."
                )
            return np.digitize(confs, self.thetas)
        else:
            raise ValueError(
                "The model is not fitted yet. Please call the fit method before."
            )

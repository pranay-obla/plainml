"""Every model plainml can train, in one table.

Adding a model means adding one ``ModelSpec`` to ``MODELS``: a key, a display name, a
factory that builds the estimator, and (optionally) a hyperparameter search space used by
``plainml tune``. Models backed by optional packages are skipped automatically when the
package isn't installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    GammaRegressor,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    RANSACRegressor,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
    TweedieRegressor,
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from plainml.errors import PlainMLError, did_you_mean, install_hint, is_installed
from plainml.tasks import CLASSIFICATION, MULTI_REGRESSION, MULTILABEL, REGRESSION

# A search-space entry: ("int", low, high, log), ("float", low, high, log) or ("cat", [choices]).
Space = dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class BuildOptions:
    seed: int = 42
    n_jobs: int = 1
    class_weight: bool = False
    pos_weight: float | None = None  # negatives / positives, for XGBoost on binary targets
    n_rows: int = 0


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Let classifiers that need 0..k-1 labels (XGBoost) accept any labels."""

    def __init__(self, estimator: Any = None):
        self.estimator = estimator

    def fit(self, X: Any, y: Any, **fit_params: Any) -> LabelEncodedClassifier:
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y), **fit_params)
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded = np.asarray(self.estimator_.predict(X)).astype(int).ravel()
        return self.encoder_.inverse_transform(encoded)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator_.feature_importances_


@dataclass(frozen=True)
class ModelSpec:
    key: str
    names: dict[str, str]  # base task -> display name
    factory: Callable[[str, BuildOptions], Any]
    description: str
    requires: str | None = None
    quick: bool = False
    max_rows: int | None = None
    native_multioutput: bool = False
    baseline: bool = False
    search: dict[str, Space] = field(default_factory=dict)  # base task -> search space
    cost: float = 1.0  # how training time grows with rows: 1 = linear, 2 = quadratic...
    tier: str = "default"  # "default": every run; "extra": only with --thorough or --models
    condition: Callable[[Any], str | None] | None = None  # returns a reason to skip, given y

    @property
    def tasks(self) -> tuple[str, ...]:
        tasks = list(self.names)
        if CLASSIFICATION in self.names:
            tasks.append(MULTILABEL)
        if REGRESSION in self.names:
            tasks.append(MULTI_REGRESSION)
        return tuple(tasks)

    @property
    def available(self) -> bool:
        return self.requires is None or is_installed(self.requires)

    def unsuitable(self, y: Any) -> str | None:
        """Why this model can't be used for this target (None if it can)."""
        return self.condition(y) if self.condition is not None and y is not None else None

    def name(self, task: str) -> str:
        return self.names[base_task(task)]

    def space(self, task: str) -> Space:
        return self.search.get(base_task(task), {})

    def build(self, task: str, options: BuildOptions) -> Any:
        estimator = self.factory(base_task(task), options)
        if task == MULTILABEL and not self.native_multioutput:
            estimator = MultiOutputClassifier(estimator)
        elif task == MULTI_REGRESSION and not self.native_multioutput:
            estimator = MultiOutputRegressor(estimator)
        return estimator


def base_task(task: str) -> str:
    return CLASSIFICATION if task in (CLASSIFICATION, MULTILABEL) else REGRESSION


def _cw(options: BuildOptions) -> str | None:
    return "balanced" if options.class_weight else None


# --- factories -------------------------------------------------------------------------


def _baseline(task: str, o: BuildOptions) -> Any:
    return (
        DummyClassifier(strategy="prior")
        if task == CLASSIFICATION
        else DummyRegressor(strategy="mean")
    )


def _linear(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return LogisticRegression(max_iter=3000, class_weight=_cw(o), random_state=o.seed)
    return LinearRegression()


def _lasso(task: str, o: BuildOptions) -> Any:
    return LassoCV(cv=5, random_state=o.seed, n_jobs=o.n_jobs, max_iter=5000)


def _elasticnet(task: str, o: BuildOptions) -> Any:
    return ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=o.seed, n_jobs=o.n_jobs, max_iter=5000
    )


def _knn(task: str, o: BuildOptions) -> Any:
    k = int(min(15, max(3, round(np.sqrt(max(o.n_rows, 1)) / 4))))
    if task == CLASSIFICATION:
        return KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=o.n_jobs)
    return KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=o.n_jobs)


def _nb(task: str, o: BuildOptions) -> Any:
    return GaussianNB()


def _svm(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return SVC(probability=True, class_weight=_cw(o), random_state=o.seed)
    return SVR()


def _tree(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return DecisionTreeClassifier(min_samples_leaf=2, class_weight=_cw(o), random_state=o.seed)
    return DecisionTreeRegressor(min_samples_leaf=2, random_state=o.seed)


def _rf(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return RandomForestClassifier(
            n_estimators=300, class_weight=_cw(o), n_jobs=o.n_jobs, random_state=o.seed
        )
    return RandomForestRegressor(n_estimators=300, n_jobs=o.n_jobs, random_state=o.seed)


def _extratrees(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return ExtraTreesClassifier(
            n_estimators=300, class_weight=_cw(o), n_jobs=o.n_jobs, random_state=o.seed
        )
    return ExtraTreesRegressor(n_estimators=300, n_jobs=o.n_jobs, random_state=o.seed)


def _gbm(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return GradientBoostingClassifier(random_state=o.seed)
    return GradientBoostingRegressor(random_state=o.seed)


def _histgb(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return HistGradientBoostingClassifier(class_weight=_cw(o), random_state=o.seed)
    return HistGradientBoostingRegressor(random_state=o.seed)


def _mlp(task: str, o: BuildOptions) -> Any:
    common: dict[str, Any] = dict(
        hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=o.seed
    )
    return MLPClassifier(**common) if task == CLASSIFICATION else MLPRegressor(**common)


def _xgboost(task: str, o: BuildOptions) -> Any:
    import xgboost as xgb

    common: dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        tree_method="hist",
        n_jobs=o.n_jobs,
        random_state=o.seed,
        verbosity=0,
    )
    if task == CLASSIFICATION:
        if o.class_weight and o.pos_weight:
            common["scale_pos_weight"] = o.pos_weight
        return LabelEncodedClassifier(xgb.XGBClassifier(**common))
    return xgb.XGBRegressor(**common)


def _lightgbm(task: str, o: BuildOptions) -> Any:
    import lightgbm as lgb

    common: dict[str, Any] = dict(
        n_estimators=300, learning_rate=0.05, n_jobs=o.n_jobs, random_state=o.seed, verbose=-1
    )
    if task == CLASSIFICATION:
        return lgb.LGBMClassifier(class_weight=_cw(o), **common)
    return lgb.LGBMRegressor(**common)


def _catboost(task: str, o: BuildOptions) -> Any:
    import catboost as cb

    common: dict[str, Any] = dict(
        iterations=500,
        random_seed=o.seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=o.n_jobs if o.n_jobs > 0 else -1,
    )
    if task == CLASSIFICATION:
        if o.class_weight:
            common["auto_class_weights"] = "Balanced"
        return cb.CatBoostClassifier(**common)
    return cb.CatBoostRegressor(**common)


def _calibrated(estimator: Any) -> Any:
    """Give a model without probabilities (linear SVM, ridge) calibrated ones."""
    return CalibratedClassifierCV(estimator, cv=3)


def _ridge_both(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return _calibrated(RidgeClassifierCV(alphas=np.logspace(-3, 3, 13), class_weight=_cw(o)))
    return RidgeCV(alphas=np.logspace(-3, 3, 13))


def _lda(task: str, o: BuildOptions) -> Any:
    return LinearDiscriminantAnalysis()


def _qda(task: str, o: BuildOptions) -> Any:
    return QuadraticDiscriminantAnalysis(reg_param=0.1)


def _adaboost(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return AdaBoostClassifier(n_estimators=200, random_state=o.seed)
    return AdaBoostRegressor(n_estimators=200, random_state=o.seed)


def _bagging(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return BaggingClassifier(n_estimators=100, n_jobs=o.n_jobs, random_state=o.seed)
    return BaggingRegressor(n_estimators=100, n_jobs=o.n_jobs, random_state=o.seed)


def _sgd(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return SGDClassifier(
            loss="log_loss", early_stopping=True, class_weight=_cw(o), random_state=o.seed
        )
    return SGDRegressor(early_stopping=True, random_state=o.seed)


def _linear_svm(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return _calibrated(LinearSVC(class_weight=_cw(o), random_state=o.seed))
    return LinearSVR(random_state=o.seed, max_iter=5000)


def _gaussian_process(task: str, o: BuildOptions) -> Any:
    if task == CLASSIFICATION:
        return GaussianProcessClassifier(random_state=o.seed)
    return GaussianProcessRegressor(normalize_y=True, random_state=o.seed)


def _bayesian_ridge(task: str, o: BuildOptions) -> Any:
    return BayesianRidge()


def _huber(task: str, o: BuildOptions) -> Any:
    return HuberRegressor(max_iter=1000)


def _theil_sen(task: str, o: BuildOptions) -> Any:
    return TheilSenRegressor(random_state=o.seed, n_jobs=o.n_jobs)


def _ransac(task: str, o: BuildOptions) -> Any:
    return RANSACRegressor(random_state=o.seed)


def _poisson(task: str, o: BuildOptions) -> Any:
    return PoissonRegressor(alpha=1e-4, max_iter=1000)


def _gamma(task: str, o: BuildOptions) -> Any:
    return GammaRegressor(alpha=1e-4, max_iter=1000)


def _tweedie(task: str, o: BuildOptions) -> Any:
    return TweedieRegressor(power=1.5, alpha=1e-4, max_iter=1000)


def _kernel_ridge(task: str, o: BuildOptions) -> Any:
    return KernelRidge(kernel="rbf")


def _pls(task: str, o: BuildOptions) -> Any:
    return PLSRegression(n_components=2)


def _torch_mlp(task: str, o: BuildOptions) -> Any:
    from plainml.deep import TorchMLPClassifier, TorchMLPRegressor

    if task == CLASSIFICATION:
        return TorchMLPClassifier(class_weight=_cw(o), random_state=o.seed)
    return TorchMLPRegressor(random_state=o.seed)


def _non_negative(y: Any) -> str | None:
    return (
        "needs a target with no negative values"
        if np.nanmin(np.asarray(y, dtype=float)) < 0
        else None
    )


def _positive(y: Any) -> str | None:
    return (
        "needs a target that is always above zero"
        if np.nanmin(np.asarray(y, dtype=float)) <= 0
        else None
    )


# --- search spaces ---------------------------------------------------------------------

_FOREST: Space = {
    "n_estimators": ("int", 100, 600, False),
    "max_depth": ("int", 3, 40, False),
    "min_samples_leaf": ("int", 1, 20, True),
    "max_features": ("cat", ["sqrt", "log2", 0.5, 1.0]),
}
_BOOSTING: Space = {
    "n_estimators": ("int", 50, 500, False),
    "learning_rate": ("float", 0.01, 0.3, True),
    "max_depth": ("int", 2, 8, False),
    "subsample": ("float", 0.5, 1.0, False),
}
_HISTGB: Space = {
    "learning_rate": ("float", 0.01, 0.3, True),
    "max_iter": ("int", 100, 800, False),
    "max_leaf_nodes": ("int", 8, 128, True),
    "min_samples_leaf": ("int", 5, 100, True),
    "l2_regularization": ("float", 1e-6, 10.0, True),
}
_XGB: Space = {
    "n_estimators": ("int", 100, 800, False),
    "learning_rate": ("float", 0.01, 0.3, True),
    "max_depth": ("int", 3, 10, False),
    "subsample": ("float", 0.5, 1.0, False),
    "colsample_bytree": ("float", 0.5, 1.0, False),
    "min_child_weight": ("float", 1.0, 10.0, True),
    "reg_lambda": ("float", 1e-3, 10.0, True),
}
_LGBM: Space = {
    "n_estimators": ("int", 100, 800, False),
    "learning_rate": ("float", 0.01, 0.3, True),
    "num_leaves": ("int", 8, 128, True),
    "min_child_samples": ("int", 5, 100, True),
    "subsample": ("float", 0.5, 1.0, False),
    "subsample_freq": ("cat", [1]),
    "colsample_bytree": ("float", 0.5, 1.0, False),
    "reg_lambda": ("float", 1e-3, 10.0, True),
}
_CATBOOST: Space = {
    "iterations": ("int", 200, 1000, False),
    "learning_rate": ("float", 0.01, 0.3, True),
    "depth": ("int", 4, 10, False),
    "l2_leaf_reg": ("float", 1.0, 10.0, True),
}
_KNN: Space = {
    "n_neighbors": ("int", 1, 50, False),
    "weights": ("cat", ["uniform", "distance"]),
    "p": ("cat", [1, 2]),
}
_TREE: Space = {"max_depth": ("int", 2, 30, False), "min_samples_leaf": ("int", 1, 50, True)}
_TORCH: Space = {
    "width": ("int", 32, 512, True),
    "depth": ("int", 1, 4, False),
    "dropout": ("float", 0.0, 0.5, False),
    "learning_rate": ("float", 1e-4, 1e-2, True),
    "weight_decay": ("float", 1e-6, 1e-2, True),
}
_ADABOOST: Space = {
    "n_estimators": ("int", 50, 500, False),
    "learning_rate": ("float", 0.01, 2.0, True),
}
_MLP: Space = {
    "alpha": ("float", 1e-6, 1e-1, True),
    "learning_rate_init": ("float", 1e-4, 1e-1, True),
}

C, R = CLASSIFICATION, REGRESSION

MODELS: list[ModelSpec] = [
    ModelSpec(
        "baseline",
        {C: "Baseline (most common class)", R: "Baseline (always the average)"},
        _baseline,
        "A do-nothing reference point. Every real model should beat it.",
        quick=True,
        baseline=True,
        native_multioutput=True,
    ),
    ModelSpec(
        "linear",
        {C: "Logistic Regression", R: "Linear Regression"},
        _linear,
        "Fits a straight-line relationship. Fast and easy to interpret.",
        quick=True,
        search={C: {"C": ("float", 1e-3, 100.0, True)}},
        native_multioutput=False,
    ),
    ModelSpec(
        "ridge",
        {C: "Ridge Classifier", R: "Ridge Regression"},
        _ridge_both,
        "A linear model that resists overfitting by keeping coefficients small.",
        quick=True,
    ),
    ModelSpec(
        "lasso",
        {R: "Lasso Regression"},
        _lasso,
        "Linear regression that can switch off unhelpful columns.",
    ),
    ModelSpec(
        "elasticnet", {R: "Elastic Net"}, _elasticnet, "A blend of ridge and lasso regression."
    ),
    ModelSpec(
        "knn",
        {C: "K-Nearest Neighbors", R: "K-Nearest Neighbors"},
        _knn,
        "Predicts from the most similar rows in the training data.",
        max_rows=200_000,
        native_multioutput=True,
        search={C: _KNN, R: _KNN},
        cost=2.0,
    ),
    ModelSpec(
        "nb", {C: "Naive Bayes"}, _nb, "A simple probabilistic classifier; very fast.", quick=True
    ),
    ModelSpec(
        "svm",
        {C: "Support Vector Machine", R: "Support Vector Machine"},
        _svm,
        "Finds the widest boundary between classes. Strong on small, clean datasets; slow on big ones.",
        max_rows=20_000,
        search={
            C: {"C": ("float", 1e-2, 100.0, True), "gamma": ("float", 1e-4, 1.0, True)},
            R: {
                "C": ("float", 1e-2, 100.0, True),
                "gamma": ("float", 1e-4, 1.0, True),
                "epsilon": ("float", 1e-3, 1.0, True),
            },
        },
        cost=2.0,
    ),
    ModelSpec(
        "tree",
        {C: "Decision Tree", R: "Decision Tree"},
        _tree,
        "A single flowchart of yes/no questions. Easy to read, prone to overfitting.",
        quick=True,
        native_multioutput=True,
        search={C: _TREE, R: _TREE},
    ),
    ModelSpec(
        "rf",
        {C: "Random Forest", R: "Random Forest"},
        _rf,
        "Hundreds of decision trees voting together. A reliable all-rounder.",
        quick=True,
        native_multioutput=True,
        search={C: _FOREST, R: _FOREST},
    ),
    ModelSpec(
        "extratrees",
        {C: "Extra Trees", R: "Extra Trees"},
        _extratrees,
        "Like a random forest with more randomness; often faster and just as good.",
        native_multioutput=True,
        search={C: _FOREST, R: _FOREST},
    ),
    ModelSpec(
        "gbm",
        {C: "Gradient Boosting", R: "Gradient Boosting"},
        _gbm,
        "Trees built one after another, each fixing the last one's mistakes.",
        max_rows=200_000,
        search={C: _BOOSTING, R: _BOOSTING},
    ),
    ModelSpec(
        "histgb",
        {C: "Histogram Gradient Boosting", R: "Histogram Gradient Boosting"},
        _histgb,
        "Fast gradient boosting for large datasets (scikit-learn's LightGBM-style booster).",
        quick=True,
        search={C: _HISTGB, R: _HISTGB},
    ),
    ModelSpec(
        "mlp",
        {C: "Neural Network (MLP)", R: "Neural Network (MLP)"},
        _mlp,
        "A small neural network. Can find complex patterns but needs more data.",
        native_multioutput=True,
        search={C: _MLP, R: _MLP},
    ),
    ModelSpec(
        "xgboost",
        {C: "XGBoost", R: "XGBoost"},
        _xgboost,
        "The classic competition-winning gradient booster.",
        requires="xgboost",
        search={C: _XGB, R: _XGB},
    ),
    ModelSpec(
        "lightgbm",
        {C: "LightGBM", R: "LightGBM"},
        _lightgbm,
        "Very fast gradient boosting; usually near the top on tabular data.",
        requires="lightgbm",
        quick=True,
        search={C: _LGBM, R: _LGBM},
    ),
    ModelSpec(
        "catboost",
        {C: "CatBoost", R: "CatBoost"},
        _catboost,
        "Gradient boosting that handles categories especially well.",
        requires="catboost",
        search={C: _CATBOOST, R: _CATBOOST},
    ),
    # --- also tried by default -------------------------------------------------------
    ModelSpec(
        "lda",
        {C: "Linear Discriminant Analysis"},
        _lda,
        "A classic statistical classifier; fast and hard to overfit.",
        quick=True,
    ),
    ModelSpec(
        "bayesian_ridge",
        {R: "Bayesian Ridge"},
        _bayesian_ridge,
        "Linear regression that tunes its own regularisation.",
        quick=True,
    ),
    ModelSpec(
        "huber",
        {R: "Huber Regression"},
        _huber,
        "Linear regression that isn't thrown off by a few extreme values.",
    ),
    # --- extra models: --thorough, or name them with --models ---------------------------
    ModelSpec(
        "qda",
        {C: "Quadratic Discriminant Analysis"},
        _qda,
        "Like LDA, but each class can have its own shape.",
        tier="extra",
    ),
    ModelSpec(
        "adaboost",
        {C: "AdaBoost", R: "AdaBoost"},
        _adaboost,
        "The original boosting method: each model focuses on the previous one's mistakes.",
        tier="extra",
        search={C: _ADABOOST, R: _ADABOOST},
    ),
    ModelSpec(
        "bagging",
        {C: "Bagged Trees", R: "Bagged Trees"},
        _bagging,
        "Many decision trees, each trained on a random resample of the rows.",
        tier="extra",
    ),
    ModelSpec(
        "sgd",
        {C: "SGD Linear Model", R: "SGD Linear Model"},
        _sgd,
        "A linear model trained in small steps; scales to very large data.",
        tier="extra",
    ),
    ModelSpec(
        "linear_svm",
        {C: "Linear SVM", R: "Linear SVM"},
        _linear_svm,
        "A support vector machine with a straight-line boundary; much faster than the kernel SVM.",
        tier="extra",
        search={
            C: {"estimator__C": ("float", 1e-3, 100.0, True)},
            R: {"C": ("float", 1e-3, 100.0, True)},
        },
    ),
    ModelSpec(
        "gaussian_process",
        {C: "Gaussian Process", R: "Gaussian Process"},
        _gaussian_process,
        "A flexible probabilistic model; excellent on small data, very slow on big data.",
        tier="extra",
        max_rows=2_000,
        cost=3.0,
    ),
    ModelSpec(
        "theil_sen",
        {R: "Theil-Sen Regression"},
        _theil_sen,
        "A linear fit based on medians; very robust to outliers.",
        tier="extra",
        max_rows=10_000,
        cost=2.0,
    ),
    ModelSpec(
        "ransac",
        {R: "RANSAC Regression"},
        _ransac,
        "Fits a line to the well-behaved rows and ignores the outliers.",
        tier="extra",
    ),
    ModelSpec(
        "poisson",
        {R: "Poisson Regression"},
        _poisson,
        "For counts (visits, orders, claims): predictions are never negative.",
        tier="extra",
        condition=_non_negative,
    ),
    ModelSpec(
        "gamma",
        {R: "Gamma Regression"},
        _gamma,
        "For positive amounts with a long tail, such as costs or durations.",
        tier="extra",
        condition=_positive,
    ),
    ModelSpec(
        "tweedie",
        {R: "Tweedie Regression"},
        _tweedie,
        "For amounts that are often exactly zero, like insurance claims.",
        tier="extra",
        condition=_non_negative,
    ),
    ModelSpec(
        "kernel_ridge",
        {R: "Kernel Ridge"},
        _kernel_ridge,
        "Ridge regression that can bend to fit curves; slow on big data.",
        tier="extra",
        max_rows=10_000,
        cost=2.0,
    ),
    ModelSpec(
        "pls",
        {R: "Partial Least Squares"},
        _pls,
        "Regression on a few combined directions; good when columns overlap a lot.",
        tier="extra",
        native_multioutput=True,
    ),
    ModelSpec(
        "torch",
        {C: "PyTorch Neural Network", R: "PyTorch Neural Network"},
        _torch_mlp,
        "A deep-learning network (PyTorch) with early stopping. Rarely beats boosting on tables, but can help in an ensemble.",
        requires="torch",
        tier="extra",
        search={C: _TORCH, R: _TORCH},
    ),
]

REGISTRY = {spec.key: spec for spec in MODELS}

_ALIASES = {
    "dummy": "baseline",
    "logreg": "linear",
    "logistic": "linear",
    "linreg": "linear",
    "lr": "linear",
    "randomforest": "rf",
    "random_forest": "rf",
    "forest": "rf",
    "et": "extratrees",
    "extra_trees": "extratrees",
    "gb": "gbm",
    "gradientboosting": "gbm",
    "hgb": "histgb",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "cat": "catboost",
    "cb": "catboost",
    "svc": "svm",
    "svr": "svm",
    "dt": "tree",
    "decision_tree": "tree",
    "neural": "mlp",
    "nn": "mlp",
    "naive_bayes": "nb",
    "enet": "elasticnet",
    "kneighbors": "knn",
    "pytorch": "torch",
    "torch_mlp": "torch",
    "deep": "torch",
    "gp": "gaussian_process",
    "ada": "adaboost",
    "linearsvm": "linear_svm",
    "linear_svc": "linear_svm",
    "br": "bayesian_ridge",
}

ENSEMBLE_KEY = "ensemble"
STACKING_KEY = "stacking"
COMBINED_KEYS = (ENSEMBLE_KEY, STACKING_KEY)


def resolve_keys(names: list[str] | None) -> list[str]:
    """Turn user-supplied model names (with aliases) into registry keys."""
    if not names:
        return []
    keys = []
    for raw in names:
        for part in str(raw).split(","):
            name = part.strip().lower().replace("-", "_")
            if not name:
                continue
            key = name if name in REGISTRY or name in COMBINED_KEYS else _ALIASES.get(name)
            if key is None:
                raise PlainMLError(
                    f"Unknown model '{part.strip()}'.{did_you_mean(name, [*REGISTRY, *_ALIASES])}",
                    hint="See all models with: plainml models",
                )
            if key not in keys:
                keys.append(key)
    return keys


@dataclass
class Selection:
    specs: list[ModelSpec]
    skipped: dict[str, str]  # key -> reason


def select_models(
    task: str,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    quick: bool = False,
    thorough: bool = False,
    n_rows: int = 0,
    y: Any = None,
) -> Selection:
    """Choose which models to try, explaining why any were left out."""
    include_keys = [k for k in resolve_keys(include) if k not in COMBINED_KEYS]
    exclude_keys = set(resolve_keys(exclude))
    specs: list[ModelSpec] = []
    skipped: dict[str, str] = {}
    candidates = [REGISTRY[k] for k in include_keys] if include_keys else MODELS
    for spec in candidates:
        if task not in spec.tasks:
            if include_keys:
                skipped[spec.key] = f"doesn't support {task}"
            continue
        if spec.key in exclude_keys:
            continue
        if not spec.available:
            if include_keys or spec.tier == "default" or thorough:
                skipped[spec.key] = f"needs '{spec.requires}' ({install_hint(spec.requires or '')})"
            continue
        reason = spec.unsuitable(y)
        if reason:
            if include_keys or spec.tier == "default" or thorough:
                skipped[spec.key] = reason
            continue
        if not include_keys:
            if quick and not spec.quick:
                continue
            if spec.tier == "extra" and not thorough:
                continue
            if spec.max_rows and n_rows > spec.max_rows:
                skipped[spec.key] = (
                    f"too slow for {n_rows:,} rows (add --models {spec.key} to force it)"
                )
                continue
        specs.append(spec)
    if include_keys and "baseline" not in [s.key for s in specs] and "baseline" not in exclude_keys:
        specs.insert(0, REGISTRY["baseline"])
    return Selection(specs=specs, skipped=skipped)


def build_stacking(task: str, members: list[tuple[str, Any]]) -> Any:
    """A small model that learns how best to combine the members' predictions."""
    if task == CLASSIFICATION:
        return StackingClassifier(
            estimators=members, final_estimator=LogisticRegression(max_iter=2000), cv=3
        )
    return StackingRegressor(
        estimators=members, final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=3
    )


def build_ensemble(task: str, members: list[tuple[str, Any]]) -> Any:
    """Average the predictions of several fitted-pipeline templates."""
    if task == CLASSIFICATION:
        return VotingClassifier(estimators=members, voting="soft")
    return VotingRegressor(estimators=members)


def estimator_param_prefix(model_step: Any) -> str:
    """Path from a pipeline's 'model' step to the underlying estimator's parameters."""
    prefix = "model__"
    current = model_step
    while isinstance(
        current, (LabelEncodedClassifier, MultiOutputClassifier, MultiOutputRegressor)
    ):
        prefix += "estimator__"
        current = current.estimator
    return prefix

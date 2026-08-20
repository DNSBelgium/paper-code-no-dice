from gamblingfinder.nn import GamblingNN
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from gamblingfinder.model import get_pipeline, FeatureConfig
from sklearn.metrics import log_loss
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import SVC

SEED = 739841


def param_binary_xgb(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 5, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 64, 256),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
    }


def param_binary_lr(
    trial, solvers=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
):
    param = {
        "solver": trial.suggest_categorical(
            "solver",
            solvers,
        ),
        "C": trial.suggest_float("C", 1e-4, 1000, log=True),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }

    if param["solver"] in (
        "lbfgs",
        "newton-cg",
        "newton-cholesky",
        "sag",
        "liblinear",
    ):
        param["l1_ratio"] = 0
    elif param["solver"] == "saga":
        param["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)
    else:
        raise ValueError(f"Unexpected solver: {param['solver']}")

    return param


def param_binary_lr_bow(trial):
    return param_binary_lr(
        trial, solvers=["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
    )


def param_multi_lr(trial):
    return param_binary_lr(
        trial, solvers=["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]
    )


def param_binary_svc(trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 1000, log=True),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "probability": trial.suggest_categorical("probability", [True]),
        "kernel": trial.suggest_categorical("kernel", ["linear"]),
    }


def param_binary_nn(trial):
    return {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 1, 20),
        "activation": trial.suggest_categorical(
            "activation", ["relu", "tanh", "sigmoid"]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 4),
    }


def generic_get_objective(X, y, model_class, param_fn, feature_config: FeatureConfig):
    def objective(trial):
        param = param_fn(trial)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = get_pipeline(model_class, feature_config, **param)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)
            scores.append(-log_loss(y_test, proba))
        score = sum(scores) / len(scores)

        return score

    return objective


def get_objective_binary_xgb(X, y):
    return generic_get_objective(
        X,
        y,
        XGBClassifier,
        param_binary_xgb,
        FeatureConfig(use_bag_of_words=0, only_use_embeddings=True),
    )


def tune_hyperparams_binary(
    ground_truth_train: pd.DataFrame, get_objective_fn
) -> pd.DataFrame:
    X = ground_truth_train[
        [col for col in ground_truth_train.columns if col.startswith("f_")]
    ]
    y = ground_truth_train["label"] > 0

    obj = get_objective_fn(X, y)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(obj, n_trials=100)
    print(study.best_trial)

    best_params = study.best_trial.params
    best_params_df = pd.DataFrame([best_params])
    return best_params_df


def tune_hyperparams_binary_xgb(ground_truth_train: pd.DataFrame) -> pd.DataFrame:
    return tune_hyperparams_binary(
        ground_truth_train, get_objective_fn=get_objective_binary_xgb
    )


def get_objective_binary_lr(
    X,
    y,
    only_use_embeddings: bool = True,
    embedding_dim: int = 384,
    embedding_feature_prefix: str = "f_emb_",
):
    return generic_get_objective(
        X,
        y,
        LogisticRegression,
        param_binary_lr,
        FeatureConfig(
            use_bag_of_words=0,
            only_use_embeddings=only_use_embeddings,
            embedding_dim=embedding_dim,
            embedding_feature_prefix=embedding_feature_prefix,
        ),
    )


def tune_hyperparams_binary_lr(
    ground_truth_train: pd.DataFrame,
    only_use_embeddings: bool = True,
    embedding_dim: int = 384,
    embedding_feature_prefix: str = "f_emb_",
) -> pd.DataFrame:
    return tune_hyperparams_binary(
        ground_truth_train,
        get_objective_fn=lambda X, y: get_objective_binary_lr(
            X, y, only_use_embeddings, embedding_dim, embedding_feature_prefix
        ),
    )


def get_objective_binary_svc(X, y, embedding_dim: int):
    return generic_get_objective(
        X,
        y,
        SVC,
        param_binary_svc,
        FeatureConfig(
            use_bag_of_words=0, only_use_embeddings=True, embedding_dim=embedding_dim
        ),
    )


def tune_hyperparams_binary_svc(
    ground_truth_train: pd.DataFrame, embedding_dim: int
) -> pd.DataFrame:
    return tune_hyperparams_binary(
        ground_truth_train,
        get_objective_fn=lambda X, y: get_objective_binary_svc(X, y, embedding_dim),
    )


def get_objective_binary_nn(X, y):
    def objective(trial):
        param = param_binary_nn(trial)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            nn = GamblingNN(**param)
            nn.train(X_train, y_train.astype(int).to_numpy())
            proba = nn.predict(X_test)
            scores.append(-log_loss(y_test, proba))
        score = sum(scores) / len(scores)

        return score

    return objective


def tune_hyperparams_binary_nn(ground_truth_train: pd.DataFrame) -> pd.DataFrame:
    return tune_hyperparams_binary(
        ground_truth_train, get_objective_fn=get_objective_binary_nn
    )


def get_objective_binary_lr_bow(X, y, tfidf: bool):
    def objective(trial):
        param = param_binary_lr_bow(trial)

        feature_config = FeatureConfig(
            use_bag_of_words=1 + int(tfidf), only_use_embeddings=None
        )
        feature_config.min_df = trial.suggest_int("min_df", 1, 20)
        feature_config.max_df = trial.suggest_float("max_df", 0.5, 1.0)
        feature_config.bow_binary = trial.suggest_categorical(
            "bow_binary", [True, False]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = get_pipeline(LogisticRegression, feature_config, **param)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)
            scores.append(-log_loss(y_test, proba))
        score = sum(scores) / len(scores)

        return score

    return objective


def tune_hyperparams_binary_lr_countvec(
    ground_truth_train: pd.DataFrame,
) -> pd.DataFrame:
    return tune_hyperparams_binary(
        ground_truth_train,
        get_objective_fn=lambda X, y: get_objective_binary_lr_bow(X, y, tfidf=False),
    )


def tune_hyperparams_binary_lr_tfidf(ground_truth_train: pd.DataFrame) -> pd.DataFrame:
    return tune_hyperparams_binary(
        ground_truth_train,
        get_objective_fn=lambda X, y: get_objective_binary_lr_bow(X, y, tfidf=True),
    )


def get_objective_multi_lr(X, y):
    return generic_get_objective(
        X,
        y,
        LogisticRegression,
        param_multi_lr,
        FeatureConfig(use_bag_of_words=0, only_use_embeddings=True),
    )


def tune_hyperparams_multi(
    ground_truth_train: pd.DataFrame, get_objective_fn
) -> pd.DataFrame:
    X = ground_truth_train[
        [col for col in ground_truth_train.columns if col.startswith("f_")]
    ]
    y = ground_truth_train["label"]

    obj = get_objective_fn(X, y)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(obj, n_trials=100)
    print(study.best_trial)

    best_params = study.best_trial.params
    best_params_df = pd.DataFrame([best_params])
    return best_params_df


def tune_hyperparams_multi_lr(ground_truth_train: pd.DataFrame) -> pd.DataFrame:
    return tune_hyperparams_multi(
        ground_truth_train, get_objective_fn=get_objective_multi_lr
    )

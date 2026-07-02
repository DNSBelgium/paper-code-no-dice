from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.metrics import f1_score
import numpy as np

SEED = 739841

NUMERIC_FEATURES = [
    "f_nb_imgs",
    "f_nb_links_int",
    "f_nb_links_ext",
    "f_nb_links_tel",
    "f_nb_links_email",
    "f_nb_input_txt",
    "f_nb_button",
    "f_nb_meta_desc",
    "f_nb_meta_keyw",
    "f_nb_numerical_strings",
    "f_nb_tags",
    "f_nb_words",
    "f_nb_letters",
    "f_html_length",
    "f_nb_facebook_shallow_links",
    "f_nb_facebook_deep_links",
    "f_nb_linkedin_deep_links",
    "f_nb_linkedin_shallow_links",
    "f_nb_twitter_deep_links",
    "f_nb_twitter_shallow_links",
    "f_nb_youtube_deep_links",
    "f_nb_youtube_shallow_links",
    "f_nb_vimeo_deep_links",
    "f_nb_vimeo_shallow_links",
    "f_nb_currency_names",
    "f_nb_distinct_currencies",
    "f_distance_title_final_dn",
    "f_distance_title_initial_dn",
    "f_longest_subsequence_title_final_dn",
    "f_longest_subsequence_title_initial_dn",
    "f_fraction_words_title_final_dn",
    "f_fraction_words_title_initial_dn",
    "f_nb_distinct_words_in_title",
    "f_nb_distinct_hosts_in_urls",
]
BOOLEAN_FEATURES = [
    "f_body_text_truncated",
    "f_meta_text_truncated",
    "f_title_truncated",
]
CATEGORICAL_FEATURES = [
    "f_body_text_language",
]
BODY_TEXT_FEATURE = "f_body_text"
META_TEXT_FEATURE = "f_meta_text"
TITLE_FEATURE = "f_title"


@dataclass
class FeatureConfig:
    use_bag_of_words: int # 0 = no bag of words, 1 = count vectorizer, 2 = tfidf vectorizer
    only_use_embeddings: bool | None
    embedding_dim: int = 384
    min_df: int | None = None
    max_df: float | None = None
    bow_binary: bool | None = None


def get_column_transformer(feature_config: FeatureConfig) -> ColumnTransformer:
    if feature_config.use_bag_of_words == 0:
        embedding_features = [f"f_emb_{i}" for i in range(1, feature_config.embedding_dim + 1)]
        transformers = (
            [
                ("num", RobustScaler(), NUMERIC_FEATURES),
                ("emb", "passthrough", embedding_features),
                ("bool", "passthrough", BOOLEAN_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
            if not feature_config.only_use_embeddings
            else [("emb", "passthrough", embedding_features)]
        )
    elif feature_config.use_bag_of_words == 1:
        transformers = [
            (
                "words",
                CountVectorizer(
                    analyzer="word",
                    min_df=feature_config.min_df,
                    max_df=feature_config.max_df,
                    binary=feature_config.bow_binary,
                    max_features=50000,
                ),
                BODY_TEXT_FEATURE,
            ),
        ]
    else:
        assert feature_config.use_bag_of_words == 2
        transformers = [
            (
                "words",
                TfidfVectorizer(
                    analyzer="word",
                    min_df=feature_config.min_df,
                    max_df=feature_config.max_df,
                    binary=feature_config.bow_binary,
                    max_features=50000,
                ),
                BODY_TEXT_FEATURE,
            ),
        ]

    return ColumnTransformer(
        transformers=transformers,
    )


def get_pipeline(model_class, feature_config: FeatureConfig, **model_params) -> Pipeline:
    model_params = model_params.copy()
    if model_class not in (SVC, LogisticRegression):
        model_params["n_jobs"] = 1  # reproducibility

    return make_pipeline(
        get_column_transformer(feature_config),
        model_class(random_state=SEED, **model_params),
    )


def save_model(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def fit(df: pd.DataFrame, pipeline: Pipeline, **fit_kwargs) -> Pipeline:
    X = df[[col for col in df.columns if col.startswith("f_")]]
    y = df["label"]

    pipeline.fit(X, y, **fit_kwargs)
    return pipeline


def exec_train_test_binary(
    ground_truth_train: pd.DataFrame,
    ground_truth_test: pd.DataFrame | None,
    params_df: pd.DataFrame,
    model_name: str,
    model_class: type,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    ground_truth_train = ground_truth_train.copy()

    ground_truth_train["label"] = ground_truth_train["label"] > 0

    params = params_df.to_dict(orient="records")[0]

    fitted = fit(
        ground_truth_train,
        get_pipeline(model_class, feature_config, **params)
    )

    save_model(fitted, f"data/{model_name}.joblib")

    if ground_truth_test is not None:
        ground_truth_test = ground_truth_test.copy()
        ground_truth_test["label"] = ground_truth_test["label"] > 0

        test_result_df = ground_truth_test[["sample_id", "label"]].copy()
        test_result_df["prediction"] = fitted.predict_proba(ground_truth_test)[:, 1]
    else:
        test_result_df = pd.DataFrame({"ok": [True]})

    return test_result_df


def exec_train_test_multi(
    ground_truth_train: pd.DataFrame,
    ground_truth_test: pd.DataFrame | None,
    params_df: pd.DataFrame,
    model_name: str,
    model_class: type,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    ground_truth_train = ground_truth_train.copy()
    ground_truth_test = ground_truth_test.copy()

    params = params_df.to_dict(orient="records")[0]

    fitted = fit(
        ground_truth_train,
        get_pipeline(model_class, feature_config, **params)
    )

    save_model(fitted, f"data/{model_name}.joblib")

    test_result_df = ground_truth_test[["sample_id", "label"]].copy()
    test_result_df["prediction"] = fitted.predict(ground_truth_test)

    probas = fitted.predict_proba(ground_truth_test)
    test_result_df["pred0"] = probas[:, 0]
    test_result_df["pred1"] = probas[:, 1]
    test_result_df["pred2"] = probas[:, 2]

    return test_result_df


def test_metrics_binary(predictions_with_features: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions_with_features["label"]
    y_score = predictions_with_features["prediction"]
    y_pred = (y_score >= 0.5).astype(int)

    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_score)

    # Precision, Recall, F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # AUC-PR (Average Precision)
    auc_pr = average_precision_score(y_true, y_score)

    # Calculate ideal threshold for F1-score
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    metrics = {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Best threshold": best_threshold,
        "Best F1-score": best_f1,
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    return metrics_df


def test_metrics_multi(labels_and_predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = labels_and_predictions["label"]
    y_pred = labels_and_predictions["prediction"]

    accuracy = (y_true == y_pred).mean()
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    f1_positive = f1_score(y_true == 1, y_pred == 1)
    precision_positive = precision_score(y_true == 1, y_pred == 1)
    recall_positive = recall_score(y_true == 1, y_pred == 1)

    f1_linksites = f1_score(y_true == 2, y_pred == 2)
    precision_linksites = precision_score(y_true == 2, y_pred == 2)
    recall_linksites = recall_score(y_true == 2, y_pred == 2)

    f1_binarized = f1_score(y_true > 0, y_pred > 0)
    precision_binarized = precision_score(y_true > 0, y_pred > 0)
    recall_binarized = recall_score(y_true > 0, y_pred > 0)

    metrics = {
        "Accuracy": accuracy,
        "Weighted F1-score": weighted_f1,
        "Macro F1-score": macro_f1,
        "F1-score (Gambling)": f1_positive,
        "Precision (Gambling)": precision_positive,
        "Recall (Gambling)": recall_positive,
        "F1-score (Link sites)": f1_linksites,
        "Precision (Link sites)": precision_linksites,
        "Recall (Link sites)": recall_linksites,
        "F1-score (Binarized)": f1_binarized,
        "Precision (Binarized)": precision_binarized,
        "Recall (Binarized)": recall_binarized,
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    return metrics_df


def predict_binary(
    features: pd.DataFrame, model_name: str, id_column: str, *extra_columns
) -> pd.DataFrame:
    fitted = load_model(f"data/{model_name}.joblib")

    result_df = features[[id_column] + list(extra_columns)].copy()
    result_df["prediction"] = fitted.predict_proba(features)[:, 1]

    return result_df


def predict_binary_and_fraction_positive(
    features: pd.DataFrame, model_name: str, id_column: str, *extra_columns
) -> tuple[pd.DataFrame, float]:
    result_df = predict_binary(features, model_name, id_column, *extra_columns)
    frac_positive = (result_df["prediction"] >= 0.5).mean()
    return result_df, frac_positive

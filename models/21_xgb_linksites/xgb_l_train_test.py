from gamblingfinder.model import exec_train_test_binary, FeatureConfig
from xgboost import XGBClassifier


def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train_linksites").df()
    ground_truth_test = dbt.ref("stg_test_linksites").df()
    params = dbt.ref("xgb_l_hyperparams").df()

    model_name = "xgb_linksites"
    result = exec_train_test_binary(
        ground_truth_train,
        ground_truth_test,
        params,
        model_name,
        XGBClassifier,
        FeatureConfig(use_bag_of_words=0, only_use_embeddings=True),
    )

    return result

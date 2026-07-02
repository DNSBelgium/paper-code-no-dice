from gamblingfinder.model import exec_train_test_binary, FeatureConfig
from sklearn.linear_model import LogisticRegression

def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train").df()
    ground_truth_test = dbt.ref("stg_test").df()
    params = dbt.ref("feat_countvec_hyperparams").df()

    model_name = "lr_f_countvec_gambling"

    feature_config = FeatureConfig(use_bag_of_words=1, only_use_embeddings=None)
    feature_config.min_df = params["min_df"].values[0]
    feature_config.max_df = params["max_df"].values[0]
    feature_config.bow_binary = params["bow_binary"].values[0]
    params.drop(columns=["min_df", "max_df", "bow_binary"], inplace=True)

    result = exec_train_test_binary(
        ground_truth_train,
        ground_truth_test,
        params,
        model_name,
        LogisticRegression,
        feature_config,
    )

    return result

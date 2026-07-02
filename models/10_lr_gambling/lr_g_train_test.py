from gamblingfinder.model import exec_train_test_binary, FeatureConfig
from sklearn.linear_model import LogisticRegression

def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train").df()
    ground_truth_test = dbt.ref("stg_test").df()
    params = dbt.ref("lr_g_hyperparams").df()

    model_name = "lr_gambling"
    result = exec_train_test_binary(
        ground_truth_train,
        ground_truth_test,
        params,
        model_name,
        LogisticRegression,
        FeatureConfig(use_bag_of_words=0, only_use_embeddings=True),
    )

    return result

from gamblingfinder.model import exec_train_test_binary, FeatureConfig
from sklearn.linear_model import LogisticRegression


def model(dbt, _session):
    ground_truth_train = dbt.ref("scr_ocr_text_a_train").df()
    ground_truth_test = dbt.ref("scr_ocr_text_a_test").df()
    params = dbt.ref("scr_ocr_text_a_hyperparams").df()

    model_name = "lr_scr_ocr_text_a_gambling"
    result = exec_train_test_binary(
        ground_truth_train,
        ground_truth_test,
        params,
        model_name,
        LogisticRegression,
        FeatureConfig(
            use_bag_of_words=0,
            only_use_embeddings=True,
            embedding_feature_prefix="f_avg_emb_",
        ),
    )

    return result

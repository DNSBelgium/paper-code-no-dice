from gamblingfinder.model import exec_train_test_binary, FeatureConfig
from sklearn.svm import SVC

def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_doc2vec_train").df()
    ground_truth_test = dbt.ref("stg_doc2vec_test").df()
    params = dbt.ref("d2v_svm_hyperparams").df()

    model_name = "d2v_svm_gambling"
    result = exec_train_test_binary(
        ground_truth_train,
        ground_truth_test,
        params,
        model_name,
        SVC,
        FeatureConfig(use_bag_of_words=0, only_use_embeddings=True, embedding_dim=200),
    )

    return result

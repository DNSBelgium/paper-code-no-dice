from gamblingfinder.tuning import tune_hyperparams_binary_svc


def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_doc2vec_train").df()
    return tune_hyperparams_binary_svc(ground_truth_train, embedding_dim=200)

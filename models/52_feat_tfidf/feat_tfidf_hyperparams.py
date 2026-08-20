from gamblingfinder.tuning import tune_hyperparams_binary_lr_tfidf


def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train").df()
    return tune_hyperparams_binary_lr_tfidf(ground_truth_train)

from gamblingfinder.tuning import tune_hyperparams_multi_lr

def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train").df()
    return tune_hyperparams_multi_lr(ground_truth_train)

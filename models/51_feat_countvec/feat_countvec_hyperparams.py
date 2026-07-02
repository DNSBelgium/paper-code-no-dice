from gamblingfinder.tuning import tune_hyperparams_binary_lr_countvec

def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train").df()
    return tune_hyperparams_binary_lr_countvec(ground_truth_train)

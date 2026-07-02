from gamblingfinder.tuning import tune_hyperparams_binary_xgb

def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train_linksites").df()
    return tune_hyperparams_binary_xgb(ground_truth_train)

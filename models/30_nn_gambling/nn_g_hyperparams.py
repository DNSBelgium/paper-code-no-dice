from gamblingfinder.tuning import tune_hyperparams_binary_nn
from gamblingfinder.nn import set_seed

def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train").df()
    set_seed()
    return tune_hyperparams_binary_nn(ground_truth_train)

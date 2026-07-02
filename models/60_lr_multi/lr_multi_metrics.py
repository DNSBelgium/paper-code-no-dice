from gamblingfinder.model import test_metrics_multi


def model(dbt, _session):
    predictions = dbt.ref("lr_multi_train_test").df()

    return test_metrics_multi(predictions)

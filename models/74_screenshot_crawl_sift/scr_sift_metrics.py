from gamblingfinder.model import test_metrics_binary


def model(dbt, _session):
    predictions = dbt.ref("scr_sift_train_test").df()

    return test_metrics_binary(predictions)

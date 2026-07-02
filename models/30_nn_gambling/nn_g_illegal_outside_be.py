from gamblingfinder.nn import GamblingNN


def model(dbt, _session):
    _ = dbt.ref("nn_g_train_test")

    features = dbt.ref("stg_illegal_outside_be").df()

    fitted = GamblingNN.load("nn_gambling")

    result_df = features[["sample_id"]].copy()
    result_df["prediction"] = fitted.predict(features)[:, 1]

    frac_positive = (result_df["prediction"] >= 0.5).mean()
    print("Recall: {:.2%}".format(frac_positive))

    return result_df

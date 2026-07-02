from gamblingfinder.model import predict_binary_and_fraction_positive


def model(dbt, _session):
    _ = dbt.ref("feat_incl_crawler_train_test")

    features = dbt.ref("stg_illegal_outside_be").df()

    df, frac = predict_binary_and_fraction_positive(features, "lr_f_incl_crawler_gambling", "sample_id")
    print("Recall: {:.2%}".format(frac))

    return df

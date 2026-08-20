from gamblingfinder.model import predict_binary_and_fraction_positive


def model(dbt, _session):
    _ = dbt.ref("scr_qwen_text_train_test")

    features = dbt.ref("stg_scr_illegal_outside_be").df()

    df, frac = predict_binary_and_fraction_positive(features, "lr_scr_qwen_text_gambling", "sample_id")
    print("Recall: {:.2%}".format(frac))

    return df

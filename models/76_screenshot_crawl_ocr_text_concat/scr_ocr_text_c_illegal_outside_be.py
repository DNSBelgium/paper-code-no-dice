from gamblingfinder.model import predict_binary_and_fraction_positive


def model(dbt, _session):
    _ = dbt.ref("scr_ocr_text_c_train_test")

    features = dbt.ref("stg_scr_illegal_outside_be").df()

    df, frac = predict_binary_and_fraction_positive(
        features, "lr_scr_ocr_text_c_gambling", "sample_id"
    )
    print("Recall: {:.2%}".format(frac))

    return df

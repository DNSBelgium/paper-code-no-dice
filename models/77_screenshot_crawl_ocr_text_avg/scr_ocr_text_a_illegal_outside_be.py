from gamblingfinder.model import predict_binary_and_fraction_positive


def model(dbt, _session):
    _ = dbt.ref("scr_ocr_text_a_train_test")

    features = dbt.ref("scr_ocr_text_a_illegal_outside_be_feat").df()

    df, frac = predict_binary_and_fraction_positive(
        features, "lr_scr_ocr_text_a_gambling", "sample_id"
    )
    print("Recall: {:.2%}".format(frac))

    return df

from gamblingfinder.model import predict_binary_and_fraction_positive


def model(dbt, _session):
    _ = dbt.ref("d2v_svm_train_test")

    features = dbt.ref("stg_doc2vec_illegal_outside_be").df()

    df, frac = predict_binary_and_fraction_positive(
        features, "d2v_svm_gambling", "sample_id"
    )
    print("Recall: {:.2%}".format(frac))

    return df

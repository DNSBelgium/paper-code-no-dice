from gamblingfinder.sift import train_test_classifier

def model(dbt, _session):
    histograms = dbt.ref("scr_sift_histograms").df()

    train = histograms[histograms.is_train]
    test = histograms[~histograms.is_train]

    X_train = train.sift_histogram.tolist()
    y_train = train.label > 0

    X_test = test.sift_histogram.tolist()
    y_test = test.label > 0

    y_pred = train_test_classifier(X_train, y_train, X_test)

    result = test[["sample_id"]].copy()
    result["label"] = y_test
    result["prediction"] = y_pred
    return result

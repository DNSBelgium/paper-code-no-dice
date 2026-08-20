from gamblingfinder.nn import GamblingNN, set_seed


def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_train").df()
    ground_truth_test = dbt.ref("stg_test").df()
    set_seed()

    model_name = "nn_gambling"
    params = dbt.ref("nn_g_hyperparams").df()
    params = params.to_dict(orient="records")[0]

    nn = GamblingNN(**params)
    nn.train(ground_truth_train, (ground_truth_train.label > 0).astype(int))
    nn.save(model_name)

    result = ground_truth_test[["sample_id"]].copy()
    result["label"] = ground_truth_test["label"] > 0
    result["prediction"] = nn.predict(ground_truth_test)[:, 1]

    return result

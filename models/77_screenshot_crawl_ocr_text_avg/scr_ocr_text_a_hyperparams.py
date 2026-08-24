from gamblingfinder.tuning import tune_hyperparams_binary_lr


def model(dbt, _session):
    ground_truth_train = dbt.ref("scr_ocr_text_a_train")
    ground_truth_train = ground_truth_train.project(
        "sample_id, label, columns('f_avg_emb_*')"
    ).df()
    return tune_hyperparams_binary_lr(
        ground_truth_train,
        only_use_embeddings=True,
        embedding_feature_prefix="f_avg_emb_",
    )

from gamblingfinder.tuning import tune_hyperparams_binary_lr

def model(dbt, _session):
    ground_truth_train = dbt.ref("stg_scr_train")
    ground_truth_train = ground_truth_train.project("sample_id, label, columns('f_qwen_img_emb_*')").df()
    return tune_hyperparams_binary_lr(
        ground_truth_train,
        only_use_embeddings=True,
        embedding_dim=2048,
        embedding_feature_prefix="f_qwen_img_emb_"
    )

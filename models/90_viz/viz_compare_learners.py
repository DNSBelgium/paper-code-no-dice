from gamblingfinder.visualize import several_precision_recall_curves, several_roc_curves
import pandas as pd


def model(dbt, _session):
    preds_xgb = dbt.ref("xgb_g_train_test").df()
    preds_lr = dbt.ref("lr_g_train_test").df()
    preds_nn = dbt.ref("nn_g_train_test").df()
    preds_d2v = dbt.ref("d2v_svm_train_test").df()

    y_trues = [
        preds_d2v["label"],
        preds_nn["label"],
        preds_xgb["label"],
        preds_lr["label"],
    ]
    y_scores = [
        preds_d2v["prediction"],
        preds_nn["prediction"],
        preds_xgb["prediction"],
        preds_lr["prediction"],
    ]
    labels = ["Doc2Vec + SVM", "NN", "XGB", "LR"]

    several_precision_recall_curves(
        y_trues,
        y_scores,
        labels,
        filename="learners_pr_curves",
    )
    several_roc_curves(
        y_trues,
        y_scores,
        labels,
        filename="learners_roc_curves",
    )

    return pd.DataFrame({"ok": [True]})

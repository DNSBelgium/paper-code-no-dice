from gamblingfinder.visualize import several_precision_recall_curves, several_roc_curves
import pandas as pd


def model(dbt, _session):
    preds_xgb = dbt.ref("xgb_l_train_test").df()
    preds_lr = dbt.ref("lr_l_train_test").df()
    preds_nn = dbt.ref("nn_l_train_test").df()

    y_trues = [
        preds_nn["label"],
        preds_lr["label"],
        preds_xgb["label"],
    ]
    y_scores = [
        preds_nn["prediction"],
        preds_lr["prediction"],
        preds_xgb["prediction"],
    ]
    labels = ["NN", "LR", "XGB"]

    several_precision_recall_curves(
        y_trues,
        y_scores,
        labels,
        filename="linksites_learners_pr_curves",
    )
    several_roc_curves(
        y_trues,
        y_scores,
        labels,
        filename="linksites_learners_roc_curves",
    )

    return pd.DataFrame({"ok": [True]})

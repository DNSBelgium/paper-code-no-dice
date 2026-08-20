from gamblingfinder.visualize import several_precision_recall_curves, several_roc_curves
import pandas as pd


def model(dbt, _session):
    preds_sift = dbt.ref("scr_sift_train_test").df()
    preds_qwen_img = dbt.ref("scr_qwen_img_train_test").df()
    preds_qwen_mm = dbt.ref("scr_qwen_mm_train_test").df()
    preds_qwen_text = dbt.ref("scr_qwen_text_train_test").df()
    preds_e5_text = dbt.ref("scr_e5_text_train_test").df()

    y_trues = [
        preds_sift["label"],
        preds_qwen_img["label"],
        preds_qwen_mm["label"],
        preds_qwen_text["label"],
        preds_e5_text["label"],
    ]
    y_scores = [
        preds_sift["prediction"],
        preds_qwen_img["prediction"],
        preds_qwen_mm["prediction"],
        preds_qwen_text["prediction"],
        preds_e5_text["prediction"],
    ]
    labels = ["SIFT + SVM", "Img. emb. (Qwen)", "Multimodal emb. (Qwen)", "Text emb. (Qwen)", "Text emb. (e5)"]

    several_precision_recall_curves(
        y_trues,
        y_scores,
        labels,
        filename="scr_classifiers_pr_curves",
        legend_font_size=14,
    )
    several_roc_curves(
        y_trues,
        y_scores,
        labels,
        filename="scr_classifiers_roc_curves",
        legend_font_size=14,
    )

    return pd.DataFrame({"ok": [True]})

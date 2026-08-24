from gamblingfinder.visualize import several_precision_recall_curves, several_roc_curves
import pandas as pd


def model(dbt, _session):
    preds_sift = dbt.ref("scr_sift_train_test").df()
    preds_qwen_img = dbt.ref("scr_qwen_img_train_test").df()
    preds_qwen_mm = dbt.ref("scr_qwen_mm_train_test").df()
    preds_qwen_text = dbt.ref("scr_qwen_text_train_test").df()
    preds_e5_text = dbt.ref("scr_e5_text_train_test").df()
    preds_ocr = dbt.ref("scr_ocr_train_test").df()
    preds_ocr_text_concat = dbt.ref("scr_ocr_text_c_train_test").df()
    preds_ocr_text_avg = dbt.ref("scr_ocr_text_a_train_test").df()

    y_trues = [
        preds_sift["label"],
        preds_qwen_img["label"],
        preds_qwen_mm["label"],
        preds_ocr["label"],
        preds_ocr_text_concat["label"],
        preds_qwen_text["label"],
        preds_ocr_text_avg["label"],
        preds_e5_text["label"],
    ]
    y_scores = [
        preds_sift["prediction"],
        preds_qwen_img["prediction"],
        preds_qwen_mm["prediction"],
        preds_ocr["prediction"],
        preds_ocr_text_concat["prediction"],
        preds_qwen_text["prediction"],
        preds_ocr_text_avg["prediction"],
        preds_e5_text["prediction"],
    ]
    labels = [
        "SIFT + SVM",
        "Img. emb. (Qwen)",
        "Multimodal emb. (Qwen)",
        "OCR emb. (e5)",
        "OCR emb. || Text emb.",
        "Text emb. (Qwen)",
        "(OCR emb. + Text emb.) / 2",
        "Text emb. (e5)",
    ]

    several_precision_recall_curves(
        y_trues,
        y_scores,
        labels,
        filename="scr_classifiers_pr_curves",
        legend_font_size=12,
    )
    several_roc_curves(
        y_trues,
        y_scores,
        labels,
        filename="scr_classifiers_roc_curves",
        legend_font_size=12,
    )

    return pd.DataFrame({"ok": [True]})

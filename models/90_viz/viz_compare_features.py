from gamblingfinder.visualize import several_precision_recall_curves, several_roc_curves
import pandas as pd

BOW_ENABLED = False


def model(dbt, _session):
    preds_embonly = dbt.ref("lr_g_train_test").df()
    preds_with_crawler_feat = dbt.ref("feat_incl_crawler_train_test").df()

    preds_countvec = None
    preds_tfidf = None
    # Uncomment below when setting BOW_ENABLED = True
    # Also requires enabling 51_feat_countvec and 52_feat_tfidf in dbt_project.yml. Note that this is not possible with the public data in data/
    # preds_countvec = dbt.ref("feat_countvec_train_test").df()
    # preds_tfidf = dbt.ref("feat_tfidf_train_test").df()

    if BOW_ENABLED:
        assert preds_countvec is not None and preds_tfidf is not None
        y_trues = [
            preds_countvec["label"],
            preds_with_crawler_feat["label"],
            preds_tfidf["label"],
            preds_embonly["label"],
        ]
        y_scores = [
            preds_countvec["prediction"],
            preds_with_crawler_feat["prediction"],
            preds_tfidf["prediction"],
            preds_embonly["prediction"],
        ]
        labels = [
            "CountVectorizer",
            "Emb. + crawler feat.",
            "TfidfVectorizer",
            "Emb. only",
        ]
    else:
        y_trues = [
            preds_with_crawler_feat["label"],
            preds_embonly["label"],
        ]
        y_scores = [
            preds_with_crawler_feat["prediction"],
            preds_embonly["prediction"],
        ]
        labels = ["Emb. + crawler feat.", "Emb. only"]

    several_precision_recall_curves(
        y_trues,
        y_scores,
        labels,
        filename="features_pr_curves",
        legend_font_size=16,
    )
    several_roc_curves(
        y_trues,
        y_scores,
        labels,
        filename="features_roc_curves",
        legend_font_size=16,
    )

    return pd.DataFrame({"ok": [True]})

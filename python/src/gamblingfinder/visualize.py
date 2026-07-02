import matplotlib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pandas as pd


def make_plots_binary(predictions_with_labels: pd.DataFrame, file_prefix: str):
    matplotlib.use("Agg")

    y_true = predictions_with_labels["label"] > 0
    y_score = predictions_with_labels["prediction"]
    y_pred = (y_score >= 0.5).astype(int)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}_roc.png")
    plt.savefig(f"plots/{file_prefix}_roc.pdf")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}_pr.png")
    plt.savefig(f"plots/{file_prefix}_pr.pdf")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}_confusion.png")
    plt.savefig(f"plots/{file_prefix}_confusion.pdf")
    plt.close()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}_calibration.png")
    plt.savefig(f"plots/{file_prefix}_calibration.pdf")
    plt.close()

    # Histogram of y_score
    plt.figure()
    plt.hist(y_score, bins=10, rwidth=0.8)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities - Test Set")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}_histogram.png")
    plt.savefig(f"plots/{file_prefix}_histogram.pdf")
    plt.close()


def make_plots_multi(predictions_with_labels: pd.DataFrame, file_prefix: str, title_appendix: str = ""):
    matplotlib.use("Agg")
    plt.rcParams.update({'font.size': 13})

    y_true = predictions_with_labels["label"]
    y_pred = predictions_with_labels["prediction"]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Pos (Gambling)", "Pos (Links)"],
    )
    disp.plot(cmap=plt.cm.Blues, colorbar=False)
    if title_appendix == "":
        plt.title("Confusion Matrix")
    else:
        plt.title(f"Confusion Matrix {title_appendix}")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}_confusion.png")
    plt.savefig(f"plots/{file_prefix}_confusion.pdf")
    plt.close()


def several_precision_recall_curves(
    y_trues, y_scores, labels: list[str], filename: str, title_appendix: str = "", legend_font_size: int | None = None,
):
    matplotlib.use("Agg")

    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(7, 6))
    for y_true, y_score, label in zip(y_trues, y_scores, labels):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        aucpr = auc(recall, precision)
        plt.plot(recall, precision, label=f"{label} (AUC={aucpr:.5f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if legend_font_size is not None:
        plt.legend(loc="lower left", reverse=True, fontsize=legend_font_size)
    else:
        plt.legend(loc="lower left", reverse=True)
    """if title_appendix == "":
        plt.title("Precision-Recall Curves")
    else:
        plt.title(f"Precision-Recall Curves {title_appendix}")"""
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.savefig(f"plots/{filename}.pdf")
    plt.close()

    plt.rcParams.update({'font.size': 10})


def several_roc_curves(
    y_trues, y_scores, labels: list[str], filename: str, title_appendix: str = "", legend_font_size: int | None = None,
):
    matplotlib.use("Agg")

    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(7, 6))
    for y_true, y_score, label in zip(y_trues, y_scores, labels):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.5f})", alpha=0.7, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if legend_font_size is not None:
        plt.legend(loc="lower right", reverse=True, fontsize=legend_font_size)
    else:
        plt.legend(loc="lower right", reverse=True)
    """
    if title_appendix == "":
        plt.title("ROC Curves")
    else:
        plt.title(f"ROC Curves {title_appendix}")
    """
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.savefig(f"plots/{filename}.pdf")
    plt.close()

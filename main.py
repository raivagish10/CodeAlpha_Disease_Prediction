
import os

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import DATASETS
from models import run_experiment, HAS_XGBOOST

OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 110


def plot_model_comparison(results_df, dataset_name):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(results_df))
    width = 0.15
    for i, metric in enumerate(metrics):
        ax.bar([p + i * width for p in x], results_df[metric], width, label=metric)
    ax.set_xticks([p + width * 2 for p in x])
    ax.set_xticklabels(results_df["Model"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {dataset_name}")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"{dataset_name}_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion(fitted, best_model_name, dataset_name):
    cm = fitted[best_model_name]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{dataset_name} — {best_model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"{dataset_name}_confusion.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_roc(fitted, dataset_name):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, info in fitted.items():
        ax.plot(info["fpr"], info["tpr"], label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {dataset_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"{dataset_name}_roc.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    all_results = []
    report_lines = [
        "# Disease Prediction — Results Report\n",
        f"Boosting model used: {'XGBoost (real)' if HAS_XGBOOST else 'HistGradientBoosting (XGBoost fallback — xgboost package unavailable offline)'}\n",
    ]

    for dataset_key, meta in DATASETS.items():
        X, y = meta["loader"]()
        pretty_name = dataset_key.replace("_", " ").title()
        data_flag = "REAL data (UCI / sklearn)" if meta["is_real"] else "SYNTHETIC data (schema-matched to UCI dataset)"

        results_df, fitted = run_experiment(X, y, pretty_name)
        all_results.append(results_df)

        comp_path = plot_model_comparison(results_df, pretty_name)
        best_model = results_df.iloc[0]["Model"]
        cm_path = plot_confusion(fitted, best_model, pretty_name)
        roc_path = plot_roc(fitted, pretty_name)

        report_lines.append(f"\n## {pretty_name}  \n*{data_flag}*  \n")
        report_lines.append(f"Samples: {len(X)}, Features: {X.shape[1]}, "
                             f"Positive class rate: {y.mean():.2%}\n")
        report_lines.append(results_df.round(4).to_markdown(index=False))
        report_lines.append(f"\n**Best model:** {best_model} "
                             f"(ROC-AUC = {results_df.iloc[0]['ROC-AUC']:.4f})\n")
        report_lines.append(f"\n![]({os.path.basename(comp_path)})\n")
        report_lines.append(f"![]({os.path.basename(cm_path)})\n")
        report_lines.append(f"![]({os.path.basename(roc_path)})\n")

        print(f"\n=== {pretty_name} ({data_flag}) ===")
        print(results_df.round(4).to_string(index=False))

    full_results = pd.concat(all_results, ignore_index=True)
    full_results.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved: {OUT_DIR}/metrics_summary.csv, {OUT_DIR}/report.md, and plots in {PLOT_DIR}/")


if __name__ == "__main__":
    main()

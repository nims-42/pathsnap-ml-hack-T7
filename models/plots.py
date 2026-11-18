"""Comparison plots, all drawn from results/*.csv. Training curves only exist for the MLP, so only the MLP gets one."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS


def bar(ax, df, cols, title, labels=None):
    x = np.arange(len(df))
    w = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(x + i * w, df[c], w, label=(labels or cols)[i])
    ax.set_xticks(x + w * (len(cols) - 1) / 2)
    ax.set_xticklabels(df["model"], rotation=20)
    ax.set_title(title)
    ax.legend()


def main():
    df = pd.read_csv(RESULTS / "model_comparison.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].bar(df["model"], df["cv_macro_f1"], yerr=df["cv_macro_f1_std"], capsize=4)
    ax[0].set_title("Cross-validation macro F1 (GroupKFold by trip)")
    ax[0].tick_params(axis="x", rotation=20)
    bar(ax[1], df, ["test_top1", "test_macro_f1"], "Student-test Top-1 and macro F1", ["Top-1", "macro F1"])
    ax[2].bar(df["model"], df["test_log_loss"], color="#d62728")
    ax[2].set_title("Student-test log loss (lower is better)")
    ax[2].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS / "plot_model_comparison.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(df["model"], df["route_continuity"])
    ax[0].set_title("Route continuity (student-test, classifier only)")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(df["model"], df["inference_ms_per_1000_pairs"])
    ax[1].set_title("Inference time (ms per 1000 rows)")
    ax[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS / "plot_continuity_inference.png", dpi=150)

    names = [n for n in df["model"] if (RESULTS / f"confusion_{n}.json").exists()]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    for a, n in zip(np.atleast_1d(axes), names):
        cm = np.array(json.load(open(RESULTS / f"confusion_{n}.json")))
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v}", ha="center", va="center", color="black", fontsize=8)
        a.set(title=n, xlabel="predicted", ylabel="true", xticks=[0, 1], yticks=[0, 1])
    plt.tight_layout()
    plt.savefig(RESULTS / "plot_confusion_matrices.png", dpi=150)

    if (RESULTS / "mlp_loss_curve.csv").exists():
        c = pd.read_csv(RESULTS / "mlp_loss_curve.csv")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(c["epoch"], c["train_loss"], label="training loss")
        ax.set(title="MLP training loss", xlabel="epoch", ylabel="log loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "plot_mlp_loss.png", dpi=150)

    if (RESULTS / "tuned_vs_untuned.csv").exists():
        t = pd.read_csv(RESULTS / "tuned_vs_untuned.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        bar(ax[0], t, ["test_top1", "test_macro_f1"], "Tuned vs untuned model (student-test)", ["Top-1", "macro F1"])
        ax[1].bar(t["model"], t["test_log_loss"], color="#d62728")
        ax[1].set_title("Log loss")
        plt.tight_layout()
        plt.savefig(RESULTS / "plot_tuned_vs_untuned.png", dpi=150)
    if (RESULTS / "optuna_trials.csv").exists():
        o = pd.read_csv(RESULTS / "optuna_trials.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(o["number"], o["value"], "o-", ms=3, label="trial")
        ax.plot(o["number"], o["value"].cummin(), label="best so far")
        ax.set(title="Optuna search (CV log loss)", xlabel="trial")
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "plot_optuna.png", dpi=150)
    print("plots written to results/")


if __name__ == "__main__":
    main()

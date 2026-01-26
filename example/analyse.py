#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


# ======================
# 全局字体设置：Times New Roman
# ======================
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'


BASE_DIR = "all"
OUT_DIR = "all_summary"


def main():
    rows = []

    # ======================
    # 遍历每个靶点目录
    # ======================
    for target in os.listdir(BASE_DIR):
        troot = os.path.join(BASE_DIR, target)
        if not os.path.isdir(troot):
            continue

        tdir = os.path.join(troot, "out")
        summary_file = os.path.join(tdir, f"{target}_explain_class_summary.csv")

        if not os.path.exists(summary_file):
            continue

        df = pd.read_csv(summary_file)

        required_cols = {"Model", "Method", "TrueLabel", "AtomMean", "AtomVar", "AtomStd"}
        if not required_cols.issubset(df.columns):
            print("skip:", summary_file)
            continue

        # 统计各项指标
        for (model, method), sub in df.groupby(["Model", "Method"]):

            if set(sub["TrueLabel"]) != {0, 1}:
                continue

            mean0 = sub[sub["TrueLabel"] == 0]["AtomMean"].mean()
            mean1 = sub[sub["TrueLabel"] == 1]["AtomMean"].mean()

            var0 = sub[sub["TrueLabel"] == 0]["AtomVar"].mean()
            var1 = sub[sub["TrueLabel"] == 1]["AtomVar"].mean()

            std0 = sub[sub["TrueLabel"] == 0]["AtomStd"].mean()
            std1 = sub[sub["TrueLabel"] == 1]["AtomStd"].mean()

            mean_diff = mean1 - mean0
            mean_diff_abs = abs(mean_diff)
            var_mean = 0.5 * (var0 + var1)
            std_mean = 0.5 * (std0 + std1)

            rows.append({
                "Target": target,
                "Model": model,
                "Method": method,
                "Mean_diff_abs": mean_diff_abs,
                "Var_mean": var_mean,
                "Std_mean": std_mean,
            })

    if not rows:
        print("NO DATA FOUND, CHECK BASE_DIR")
        return

    full = pd.DataFrame(rows)

    summary = (
        full.groupby(["Model", "Method"])[["Mean_diff_abs", "Var_mean", "Std_mean"]]
        .mean()
        .reset_index()
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    full.to_csv(os.path.join(OUT_DIR, "all_targets_raw.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "all_targets_summary.csv"), index=False)

    print(summary)

    draw_plots(full, summary, OUT_DIR)


# ======================
# 绘图函数（无 legend）
# ======================
def draw_plots(full, summary, out_dir):

    sns.set(style="whitegrid")
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    COLORS = {
        "attention": "#f57c6e",
        "gnnexplainer": "#BFDFD2"
    }

    full_plot = full.copy()
    summary_plot = summary.copy()

    full_plot["Model_Method"] = full_plot["Model"] + "_" + full_plot["Method"]
    summary_plot["Model_Method"] = summary_plot["Model"] + "_" + summary_plot["Method"]

    metrics = {
        "Mean_diff_abs": "mean diff (abs)",
        "Var_mean": "variance",
        "Std_mean": "std",
    }

    # ============ 1. barplot（无 legend） ============
    for metric, label in metrics.items():
        plt.figure(figsize=(7, 5))
        sns.barplot(
            data=summary_plot,
            x="Model_Method",
            y=metric,
            hue="Method",
            palette=COLORS
        )
        plt.title(f"{label} over targets")
        plt.xlabel("Model × Method")
        plt.ylabel(label)
        plt.legend().remove()     # 去掉 legend
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"bar_{metric}.png"), dpi=600)
        plt.close()

    # ============ 2. boxplot（无 legend） ============
    for metric, label in metrics.items():
        plt.figure(figsize=(7, 5))
        sns.boxplot(
            data=full_plot,
            x="Model_Method",
            y=metric,
            hue="Method",
            palette=COLORS
        )
        plt.title(label + " (boxplot)")
        plt.xlabel("Model × Method")
        plt.ylabel(label)
        plt.legend().remove()     # 去掉 legend
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_{metric}.png"), dpi=600)
        plt.close()

    # ============ 3. violinplot（无 legend） ============
    for metric, label in metrics.items():
        plt.figure(figsize=(7, 5))
        sns.violinplot(
            data=full_plot,
            x="Model_Method",
            y=metric,
            hue="Method",
            palette=COLORS,
            inner="box",
            cut=0,
        )
        plt.title(label + " (violin)")
        plt.xlabel("Model × Method")
        plt.ylabel(label)
        plt.legend().remove()     # 去掉 legend
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"violin_{metric}.png"), dpi=600)
        plt.close()

    print("Figures saved to", out_dir)


if __name__ == "__main__":
    main()

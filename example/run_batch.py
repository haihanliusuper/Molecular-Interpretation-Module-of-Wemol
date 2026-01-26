#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import csv
import re


def run_cmd(cmd, cwd):
    print(f"\n>>> 在 {cwd} 运行: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        encoding="utf-8",     # 强制按 utf-8 读取
        errors="ignore",      # 遇到非法字符直接跳过
        text=True,
        capture_output=True
    )
    print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)
    return result.returncode, result.stdout, result.stderr



def parse_metrics_from_stdout(stdout):
    """
    从 训练dude.py 的输出中解析
    [GIN] Final best model on test set | Accuracy: 0.9123 | AUC: 0.9456
    这样的行
    """
    metrics = {}
    pattern = re.compile(
        r"\[(?P<model>\w+)\]\s+Final best model on test set \|\s+Accuracy:\s+(?P<acc>[0-9.]+)\s+\|\s+AUC:\s+(?P<auc>[0-9.NA/]+)"
    )

    for line in stdout.splitlines():
        m = pattern.search(line)
        if m:
            model = m.group("model")
            acc = float(m.group("acc"))
            auc_str = m.group("auc")
            try:
                auc = float(auc_str)
            except ValueError:
                auc = None
            metrics[model] = (acc, auc)
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="批量跑 DUD-E 所有靶点: save_data.py + 训练dude.py + 画图.py"
    )
    ap.add_argument(
        "--all_dir",
        type=str,
        default="all",
        help="DUD-E 所有靶点目录 里面是一堆 aa2ar 这类子文件夹"
    )
    ap.add_argument(
        "--save_data_script",
        type=str,
        default="save_data.py",
        help="save_data.py 路径"
    )
    ap.add_argument(
        "--train_script",
        type=str,
        default="训练dude.py",
        help="训练 GIN/GAT 的脚本路径"
    )
    ap.add_argument(
        "--explain_script",
        type=str,
        default="画图.py",
        help="画图和解释性的脚本路径"
    )
    ap.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="测试集比例 传给 save_data.py"
    )
    ap.add_argument(
        "--summary_csv",
        type=str,
        default="dude_targets_summary.csv",
        help="最终统计表输出路径"
    )
    ap.add_argument(
        "--expl_epochs",
        type=int,
        default=50,
        help="传给画图.py 的 expl_epochs"
    )
    ap.add_argument(
        "--num_per_label",
        type=int,
        default=5,
        help="画图时每个标签抽取的分子数量"
    )
    args = ap.parse_args()

    base_dir = os.path.abspath(args.all_dir)
    save_data_script = os.path.abspath(args.save_data_script)
    train_script = os.path.abspath(args.train_script)
    explain_script = os.path.abspath(args.explain_script)

    print(f"all_dir        = {base_dir}")
    print(f"save_data.py   = {save_data_script}")
    print(f"训练脚本        = {train_script}")
    print(f"画图脚本        = {explain_script}")
    print(f"summary_csv    = {os.path.abspath(args.summary_csv)}")

    rows = []

    # 遍历所有靶点目录
    for target_name in sorted(os.listdir(base_dir)):
        target_dir = os.path.join(base_dir, target_name)
        if not os.path.isdir(target_dir):
            continue

        actives_path = os.path.join(target_dir, "actives_final.ism")
        decoys_path = os.path.join(target_dir, "decoys_final.ism")

        if not (os.path.exists(actives_path) and os.path.exists(decoys_path)):
            print(f"\n### 跳过 {target_name}: 缺少 actives_final.ism 或 decoys_final.ism")
            continue

        print("\n" + "=" * 80)
        print(f"### 处理靶点: {target_name}")
        print("=" * 80)

        # 1) 生成 dude_train.pt / dude_test.pt
        cmd_save = [
            "python",
            save_data_script,
            "--actives", actives_path,
            "--decoys", decoys_path,
            "--test_size", str(args.test_size),
            "--output_prefix", "dude"
        ]
        code, out, err = run_cmd(cmd_save, cwd=target_dir)
        if code != 0:
            print(f"!!! {target_name} save_data.py 失败 跳过")
            continue

        train_pt = os.path.join(target_dir, "dude_train.pt")
        test_pt = os.path.join(target_dir, "dude_test.pt")

        if not (os.path.exists(train_pt) and os.path.exists(test_pt)):
            print(f"!!! {target_name} 缺少 dude_train.pt 或 dude_test.pt 跳过")
            continue

        # 2) 训练 GAT 和 GIN
        pdf_name = f"{target_name}_gat_gin_report.pdf"
        pdf_path = os.path.join(target_dir, pdf_name)

        cmd_train = [
            "python",
            train_script,
            "--train_pt", train_pt,
            "--test_pt", test_pt,
            "--output_pdf", pdf_path
        ]
        code, train_out, train_err = run_cmd(cmd_train, cwd=target_dir)
        if code != 0:
            print(f"!!! {target_name} 训练阶段失败 跳过该靶点的指标记录和画图")
            continue

        # 解析训练输出里的指标
        metrics = parse_metrics_from_stdout(train_out)
        if not metrics:
            print(f"!!! {target_name} 没找到最终指标 请检查训练输出格式")
        else:
            for model_name, (acc, auc) in metrics.items():
                rows.append({
                    "target": target_name,
                    "model": model_name,
                    "accuracy": acc,
                    "auc": auc
                })
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                print(f"==> {target_name} / {model_name} | Acc = {acc:.4f} | AUC = {auc_str}")

        # 3) 解释性画图
        gin_weight = "best_GIN_model.pth"
        gat_weight = "best_GAT_model.pth"
        if not (os.path.exists(os.path.join(target_dir, gin_weight))
                and os.path.exists(os.path.join(target_dir, gat_weight))):
            print(f"!!! {target_name} 缺少 best_GIN_model.pth 或 best_GAT_model.pth 画图跳过")
            continue

        explain_prefix = f"{target_name}_explain"

        cmd_explain = [
            "python",
            explain_script,
            "--test_pt", test_pt,
            "--gin_weight", gin_weight,
            "--gat_weight", gat_weight,
            "--expl_epochs", str(args.expl_epochs),
            "--num_per_label", str(args.num_per_label),
            "--output_prefix", explain_prefix
        ]
        code, exp_out, exp_err = run_cmd(cmd_explain, cwd=target_dir)
        if code != 0:
            print(f"!!! {target_name} 画图阶段失败 但训练结果仍已记录")


    # 4) 汇总各靶点指标
    if rows:
        summary_path = os.path.abspath(args.summary_csv)
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["target", "model", "accuracy", "auc"])
            writer.writeheader()
            writer.writerows(rows)
        print("\n✅ 所有靶点处理完成 结果写入:", summary_path)
    else:
        print("\n⚠️ 没有成功记录到任何靶点的指标 请检查脚本输出")


if __name__ == "__main__":
    main()

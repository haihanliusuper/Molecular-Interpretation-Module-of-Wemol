#!/root/miniconda3/envs/reinvent4/bin/python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import pandas as pd
import textwrap
import pathlib
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
# -----------------------------------------------------------------------------#
# Helper
# -----------------------------------------------------------------------------#
def run(cmd: list[str], step: str):
    print(f"\n🚀  {step}\n   {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"✅  {step} 完成\n")

# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="ChemProp + REINVENT one-stop pipeline",
)

parser.add_argument("--mode", choices=["reinvent", "libinvent", "linkinvent",
                                       "mol2mol", "pepinvent"], default="mol2mol")
parser.add_argument("--input_csv", default="input.csv")
parser.add_argument("--input_smi", default="input.smi")
parser.add_argument("--checkpoint_dir", default="checkpoint")
parser.add_argument("--toml_out", default=None)
parser.add_argument("--reinvent_log", default="sampling.log")
parser.add_argument("--sample_strategy", default="multinomial")
parser.add_argument("--distance_threshold", type=int, default=1000)
parser.add_argument("--termination", choices=["simple", "convergence"], default="simple",
                    help="终止策略：simple 或 convergence（自动判断收敛）")
parser.add_argument("--chemprop_epochs", type=int, default=10,
                    help="ChemProp 训练周期数（默认：10）")
parser.add_argument("--replace_index", type=int,
                    help="仅对 libinvent 模式生效：替换分子中的某个原子为 [*]")

args = parser.parse_args()

if args.toml_out is None:
    args.toml_out = f"generated_config_{args.mode}.toml"

out_toml = pathlib.Path(args.toml_out)
ckpt_dir = pathlib.Path(args.checkpoint_dir)

# -----------------------------------------------------------------------------#
# libinvent 专用：自动替换 SMILES 中指定的氢原子为 [*]，并保存前后图片
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# libinvent 专用：自动替换 SMILES 中指定的氢原子为 [*]，并保存前后图片
# -----------------------------------------------------------------------------#
if args.mode == "libinvent":
    print(f"🔍 Detected mode: libinvent — auto replace atom with [*] and save images")

    # 读取输入 SMILES 文件
    with open(args.input_smi, 'r') as f:
        line = f.readline().strip()
        smiles = line.split()[0]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sys.exit(f"❌ Cannot parse SMILES: {smiles}")

    # 关键：显式添加氢
    mol = Chem.AddHs(mol)

    print(f"SMILES: {smiles}  |  NumAtoms (with explicit H): {mol.GetNumAtoms()}")

    if args.replace_index is None:
        sys.exit("❌ You must provide --replace_index for libinvent mode.")

    atom_idx = args.replace_index
    if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
        sys.exit(f"❌ Invalid replace_index: {atom_idx} (valid range: 0 to {mol.GetNumAtoms()-1})")

    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "H":
        sys.exit(f"❌ Only hydrogen atoms can be replaced. Atom #{atom_idx} is {atom.GetSymbol()}.")

    # 保存原分子带原子编号的图
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    with open("libinvent_original.png", 'wb') as f:
        f.write(drawer.GetDrawingText())
    print("✅ Saved: libinvent_original.png")

    # 替换为 [*]
    emol = Chem.EditableMol(mol)
    emol.ReplaceAtom(atom_idx, Chem.Atom(0))  # wildcard [*]
    modified_mol = emol.GetMol()
    Chem.SanitizeMol(modified_mol)
    modified_smiles = Chem.MolToSmiles(modified_mol).replace("*", "[*]")

    # 保存修改后分子带原子编号的图
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(modified_mol)
    drawer.FinishDrawing()
    with open("libinvent_modified.png", 'wb') as f:
        f.write(drawer.GetDrawingText())
    print("✅ Saved: libinvent_modified.png")

    # 保存到新的 SMILES 文件，不覆盖原文件
    new_smi_file = pathlib.Path(args.input_smi).with_stem(
        pathlib.Path(args.input_smi).stem + "_libinvent"
    )
    with open(new_smi_file, 'w') as f:
        f.write(modified_smiles + "\n")

    print(f"✅ libinvent: Atom #{atom_idx} (H) replaced → New SMILES saved to {new_smi_file}")

    # 覆盖参数，让后续步骤使用新的 SMILES 文件
    args.input_smi = str(new_smi_file)


# -----------------------------------------------------------------------------#
# 1. ChemProp 训练
# -----------------------------------------------------------------------------#
run(
    ["chemprop_train",
     "--data_path", args.input_csv,
     "--dataset_type", "regression",
     "--save_dir", str(ckpt_dir),
     "--epochs", str(args.chemprop_epochs)],
    "ChemProp 训练",
)
# -----------------------------------------------------------------------------#
# 2. 计算统计量
# -----------------------------------------------------------------------------#
df = pd.read_csv(args.input_csv)
label = df["label"]
min_val = label.min()
mean_val = label.mean()
p05 = label.quantile(0.05)
p10 = label.quantile(0.10)
p005 = label.quantile(0.005)  # 前0.5%分位点，剔除极端低值

stages = [
    dict(name="Stage 1", chkpt="stage1.chkpt", max_score=0.2,
         min_steps=200,  max_steps=350,
         chemprop_high=round(mean_val +1, 2),
         chemprop_low =round(p005, 2),  k=1),

    dict(name="Stage 2", chkpt="stage2.chkpt", max_score=0.2,
         min_steps=200,  max_steps=1000,
         chemprop_high=round(mean_val-1, 2),
         chemprop_low =round(p005, 2),  k=1.0),

    dict(name="Stage 3", chkpt="stage3.chkpt", max_score=0.2,
         min_steps=200, max_steps=1000,
         chemprop_high=round(p05+1, 2),
         chemprop_low =round(p005, 2), k=1.2),

    dict(name="Stage 4", chkpt="stage4.chkpt", max_score=0.5,
         min_steps=200, max_steps=2000,
         chemprop_high=round(p05, 2),
         chemprop_low =round(p005 -1, 2), k=1.2),
]

qed_tf = {"high": 1.2, "low": 0.4, "k": 0.4, "weight": 0.3}

# -----------------------------------------------------------------------------#
# 3. 模式选择
# -----------------------------------------------------------------------------#
prior_base = "/mnt/e/shanda/wemol_example/chucun/REINVENT4-main/qianghua_test/shap2/priors"

mode2prior = {
    "reinvent":  (f"{prior_base}/reinvent.prior",      False),
    "libinvent": (f"{prior_base}/libinvent.prior",     True),
    "linkinvent":(f"{prior_base}/linkinvent.prior",    True),
    "mol2mol":   (f"{prior_base}/mol2mol_medium_similarity.prior", True),
    "pepinvent": (f"{prior_base}/pepinvent.prior",     True),
}

prior_file, need_smiles = mode2prior[args.mode]
agent_file = prior_file

# -----------------------------------------------------------------------------#
# 4. 写 TOML 文件 （此处与原始一致，未动）
# -----------------------------------------------------------------------------#
header = f"""run_type = "staged_learning"
use_cuda = true
tb_logdir = "tb_logs"
json_out_config = "_staged_learning.json"

[parameters]
summary_csv_prefix = "staged_learning"
use_checkpoint = true
purge_memories = false

prior_file  = "{prior_file}"
agent_file  = "{agent_file}"
"""

if need_smiles:
    header += f'smiles_file = "{args.input_smi}"\n'

if args.mode in {"linkinvent", "mol2mol", "pepinvent"}:
    header += f'sample_strategy = "{args.sample_strategy}"\n'
    header += f'distance_threshold = {args.distance_threshold}\n'

header += textwrap.dedent(f"""
batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001
""")

header += textwrap.dedent("""
[diversity_filter]
type              = "IdenticalMurckoScaffold"
bucket_size       = 100
minscore          = 0.4
penalty_multiplier = 0.8
""")

with open(out_toml, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for st in stages:
        f.write(f"# {st['name']}\n[[stage]]\n")
        f.write(f"chkpt_file = \"{st['chkpt']}\"\n")
        if args.termination == "convergence":
            f.write("termination = \"convergence\"\n")
            f.write("window_size = 100\n")
            f.write("std_threshold = 0.01\n")
        else:
            f.write("termination = \"simple\"\n")
        f.write(f"max_score = {st['max_score']}\n")
        f.write(f"min_steps = {st['min_steps']}\n")
        f.write(f"max_steps = {st['max_steps']}\n\n")

        f.write("[stage.scoring]\n")
        f.write("type = \"geometric_mean\"\n\n")

        f.write("[[stage.scoring.component]]\n")
        f.write("[stage.scoring.component.ChemProp]\n\n")
        f.write("[[stage.scoring.component.ChemProp.endpoint]]\n")
        f.write("name = \"ChemProp\"\nweight = 1.0\n\n")
        f.write("params.checkpoint_dir = \"" + str(ckpt_dir) + "\"\n")
        f.write("params.rdkit_2d_normalized = false\n")
        f.write("params.target_column = \"label\"\n\n")
        f.write("transform.type = \"reverse_sigmoid\"\n")
        f.write(f"transform.high = {st['chemprop_high']}\n")
        f.write(f"transform.low  = {st['chemprop_low']}\n")
        f.write(f"transform.k    = {st['k']}\n\n")

        f.write("[[stage.scoring.component]]\n")
        f.write("[stage.scoring.component.QED]\n")
        f.write("[[stage.scoring.component.QED.endpoint]]\n")
        f.write(f"name = \"QED\"\nweight = {qed_tf['weight']}\n")
        f.write("transform.type = \"sigmoid\"\n")
        for key in ("high", "low", "k"):
            f.write(f"transform.{key} = {qed_tf[key]}\n")
        f.write("\n")

print(f"✅  TOML 已保存 → {out_toml}")

# -----------------------------------------------------------------------------#
# 5. 启动 REINVENT
# -----------------------------------------------------------------------------#
run(
    ["reinvent", "-l", args.reinvent_log, str(out_toml)],
    f"REINVENT ({args.mode})",
)

# -----------------------------------------------------------------------------#
# 6. 合并输出 CSV 文件，仅保留关键列
# -----------------------------------------------------------------------------#
import glob

print("🔍 合并输出文件...")

# 匹配实际生成的文件名，例如 staged_learning_1.csv、staged_learning_2.csv
csv_files = sorted(glob.glob("staged_learning_*.csv"))

if not csv_files:
    print("⚠️  未找到任何输出文件，跳过合并。")
    sys.exit(0)

all_data = []

for file in csv_files:
    try:
        stage_num = pathlib.Path(file).stem.split("_")[2]  # 例如 1, 2, 3
        stage_name = f"stage{stage_num}"
    except IndexError:
        print(f"⚠️  文件名格式无法解析阶段: {file}，跳过。")
        continue

    df = pd.read_csv(file)

    # 检查需要的列是否存在
    required_cols = ["SMILES", "ChemProp (raw)", "QED (raw)", "Scaffold"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠️  文件 {file} 缺少列: {missing}，跳过。")
        continue

    df = df[required_cols].copy()
    df.insert(0, "Stage", stage_name)
    all_data.append(df)

if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_csv = "merged_output_summary.csv"
    merged_df.to_csv(merged_csv, index=False)
    print(f"✅ 已保存合并文件 → {merged_csv}")
else:
    print("❌ 未能生成合并文件，所有文件缺失必要列。")

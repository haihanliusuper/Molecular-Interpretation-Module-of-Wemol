#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Dropout, ReLU
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, GATConv
from torch_scatter import scatter

from torch_geometric.explain import Explainer, GNNExplainer

from rdkit import Chem
from rdkit.Chem import GetAdjacencyMatrix
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.cm as cm
import matplotlib.colors as mcolors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
#  特征 & 工具函数
# ============================

def one_hot_encoding(value, choices):
    return [1 if value == choice else 0 for choice in choices]


def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Unknown']
    if not hydrogens_implicit:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    degree_enc = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridization_enc = one_hot_encoding(
        str(atom.GetHybridization()),
        ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
    )
    is_in_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]

    pt = Chem.GetPeriodicTable()
    vdw_radius_scaled = [(pt.GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6]
    covalent_radius_scaled = [(pt.GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76]

    atom_feature_vector = (
        atom_type_enc
        + degree_enc
        + formal_charge_enc
        + hybridization_enc
        + is_in_ring_enc
        + is_aromatic_enc
        + vdw_radius_scaled
        + covalent_radius_scaled
    )

    if use_chirality:
        chirality_type_enc = one_hot_encoding(
            str(atom.GetChiralTag()),
            ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW",
             "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
        )
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4, "MoreThanFour"],
        )
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector, dtype=np.float32)


def get_bond_features(bond, use_stereochemistry=True):
    bond_type_enc = one_hot_encoding(
        bond.GetBondType(),
        [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
    )
    is_conj_enc = [int(bond.GetIsConjugated())]
    is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + is_conj_enc + is_in_ring_enc

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(
            str(bond.GetStereo()),
            ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"],
        )
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector, dtype=np.float32)


def load_test_pt(path):
    obj = torch.load(path, map_location="cpu")
    data_list = obj["data_list"]

    if "x_smiles" in obj:
        smiles = obj["x_smiles"]
    elif "smiles" in obj:
        smiles = obj["smiles"]
    else:
        smiles = []
        for d in data_list:
            if hasattr(d, "smiles"):
                smiles.append(d.smiles)
            else:
                smiles.append(None)

    labels = np.array([int(d.y.item()) for d in data_list], dtype=int)
    return data_list, smiles, labels


def select_indices_by_label(labels, num_per_label=5):
    idx_list = []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        idx_list.extend(idx_c[:num_per_label].tolist())
    return idx_list


# ============================
#  Attention Pooling + 模型
# ============================

class AttentionPooling(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.attention_mechanism = nn.Linear(node_dim, 1)
        self.mask_layer = nn.Linear(node_dim, 1)

    def forward(self, node_feats, batch_idx):
        attn_scores = self.attention_mechanism(node_feats)   # [N, 1]
        mask_logits = self.mask_layer(node_feats)            # [N, 1]
        node_mask = torch.sigmoid(mask_logits)               # [N, 1]

        final_scores = attn_scores * node_mask               # [N, 1]
        pooled = scatter(node_feats * final_scores, batch_idx, dim=0, reduce="sum")

        return pooled, final_scores.squeeze(-1)              # pooled: [B, d], AtomScores: [N]


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2,
                 out_dim, dropout_rate=0.5):
        super().__init__()
        self.mlp1 = nn.Sequential(
            Linear(in_dim, hidden_dim1),
            ReLU(),
            Linear(hidden_dim1, hidden_dim1),
        )
        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = nn.Sequential(
            Linear(hidden_dim1, hidden_dim2),
            ReLU(),
            Linear(hidden_dim2, hidden_dim2),
        )
        self.conv2 = GINConv(self.mlp2)

        self.dropout = Dropout(dropout_rate)
        self.attention_pooling = AttentionPooling(hidden_dim2)
        self.out = Linear(hidden_dim2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x, attn = self.attention_pooling(x, batch)
        return self.out(x), attn


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2,
                 out_dim, dropout_rate=0.5, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim1, heads=heads)
        self.conv2 = GATConv(hidden_dim1 * heads, hidden_dim2, heads=1)
        self.dropout = Dropout(dropout_rate)
        self.attention_pooling = AttentionPooling(hidden_dim2)
        self.out = Linear(hidden_dim2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x, attn = self.attention_pooling(x, batch)
        return self.out(x), attn


def load_model(ModelClass, weight_path, in_dim, h1, h2, out_dim, drop):
    m = ModelClass(in_dim, h1, h2, out_dim, drop)
    state = torch.load(weight_path, map_location=device)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m


def predict(model, data_list, task_type="classification"):
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list(data_list).to(device)
        out = model(batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        if task_type == "classification":
            prob = F.softmax(logits, dim=1)[:, 1]
            scores = prob.cpu().numpy()
            labels = (prob >= 0.5).int().cpu().numpy()
            return scores, labels
        else:
            scores = logits.view(-1).cpu().numpy()
            return scores, None


# ============================
#  Explainer 包装
# ============================

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch):
        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = self.model(data)
        return out[0] if isinstance(out, (tuple, list)) else out


# ============================
#  Attention 热图
# ============================

def save_attention_images(smiles_list, models, data_list,
                          base_dir="images_attention_test"):
    for model_name, model in models.items():
        model.eval()
        dev = next(model.parameters()).device
        out_dir = os.path.join(base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        for idx, (smi, data) in enumerate(zip(smiles_list, data_list)):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"[{model_name}] idx={idx} SMILES 无效，跳过")
                continue

            d = copy.deepcopy(data)
            batch = Batch.from_data_list([d]).to(dev)

            with torch.no_grad():
                out = model(batch)
                if not (isinstance(out, (tuple, list)) and len(out) == 2):
                    print(f"[{model_name}] 未返回 attn，跳过 idx={idx}")
                    continue
                _, attn = out

            attn = attn.cpu().numpy().flatten()
            if mol.GetNumAtoms() != len(attn):
                print(f"[{model_name}] idx={idx} 原子数不匹配，跳过")
                continue

            vmin, vmax = attn.min(), attn.max()
            vcenter = 0.0
            eps = 1e-6
            if vmin >= vcenter:
                vmin = vcenter - eps
            if vmax <= vcenter:
                vmax = vcenter + eps

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            cmap = cm.get_cmap("coolwarm")

            highlight_colors = {
                i: tuple(cmap(norm(w))[:3]) for i, w in enumerate(attn)
            }
            highlight_radii = {i: 0.5 for i in range(len(attn))}

            drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_colors.keys()),
                highlightAtomColors=highlight_colors,
                highlightAtomRadii=highlight_radii,
            )
            drawer.FinishDrawing()

            img_path = os.path.join(out_dir, f"mol_{idx}.png")
            with open(img_path, "wb") as f:
                f.write(drawer.GetDrawingText())

    print(f"🧠 Attention 热图已保存至 {base_dir}/<Model>/mol_X.png")


# ============================
#  GNNExplainer 热图
# ============================

def save_gnnexplainer_images(smiles_list, models, data_list,
                             task_type="classification",
                             base_dir="images_explainer_test",
                             expl_epochs=200):
    for model_name, model in models.items():
        dev = next(model.parameters()).device
        out_dir = os.path.join(base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        model_cfg = dict(
            mode='multiclass_classification' if task_type == 'classification' else 'regression',
            task_level='graph',
            return_type='log_probs' if task_type == 'classification' else 'raw',
        )

        explainer = Explainer(
            model=WrappedModel(model),
            algorithm=GNNExplainer(epochs=expl_epochs),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=model_cfg,
        )

        for idx, (smi, data) in enumerate(zip(smiles_list, data_list)):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"[{model_name}] idx={idx} SMILES 无效，跳过")
                continue

            d = copy.deepcopy(data)
            d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
            d = d.to(dev)

            try:
                explanation = explainer(
                    x=d.x,
                    edge_index=d.edge_index,
                    batch=d.batch,
                )
            except Exception as e:
                print(f"[{model_name}] idx={idx} GNNExplainer 失败：{e}")
                continue

            node_mask = explanation.get("node_mask")
            if node_mask is None:
                print(f"[{model_name}] idx={idx} 无 node_mask，跳过")
                continue

            score = node_mask.cpu().numpy().flatten()
            if mol.GetNumAtoms() != len(score):
                print(f"[{model_name}] idx={idx} 原子数不匹配，跳过")
                continue

            vmin, vmax = score.min(), score.max()
            vcenter = 0.0
            eps = 1e-6
            if vmin >= vcenter:
                vmin = vcenter - eps
            if vmax <= vcenter:
                vmax = vcenter + eps

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            cmap = cm.get_cmap("coolwarm")

            highlight_colors = {
                i: tuple(cmap(norm(s))[:3]) for i, s in enumerate(score)
            }
            highlight_radii = {i: 0.5 for i in range(len(score))}

            drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_colors.keys()),
                highlightAtomColors=highlight_colors,
                highlightAtomRadii=highlight_radii,
            )
            drawer.FinishDrawing()

            img_path = os.path.join(out_dir, f"mol_{idx}.png")
            with open(img_path, "wb") as f:
                f.write(drawer.GetDrawingText())

    print(f"🎯 GNNExplainer 热图已保存至 {base_dir}/<Model>/mol_X.png")


# ============================
#  统计：AtomMean / AtomVar / AtomStd
# ============================

def collect_attention_stats(smiles_list, models, data_list,
                            true_labels, pred_scores_dict, pred_labels_dict,
                            task_type="classification"):
    stat_rows = []

    for model_name, model in models.items():
        model.eval()
        dev = next(model.parameters()).device

        scores = pred_scores_dict[model_name]
        preds = pred_labels_dict.get(model_name, None)

        for idx, (smi, data) in enumerate(zip(smiles_list, data_list)):
            d = copy.deepcopy(data)
            batch = Batch.from_data_list([d]).to(dev)

            with torch.no_grad():
                out = model(batch)
                if not (isinstance(out, (tuple, list)) and len(out) == 2):
                    continue
                _, attn = out

            attn = attn.cpu().numpy().flatten()
            n_atoms = len(attn)

            row = {
                "Smiles":     smi,
                "MolIndex":   idx,
                "Model":      model_name,
                "Method":     "attention",
                "NumAtoms":   int(n_atoms),
                "AtomMean":   float(attn.mean()),
                "AtomVar":    float(attn.var()),
                "AtomStd":    float(attn.std()),
                "TrueLabel":  int(true_labels[idx]),
                "PredScore":  float(scores[idx]),
            }
            if task_type == "classification" and preds is not None:
                row["PredLabel"] = int(preds[idx])
            else:
                row["PredLabel"] = np.nan

            stat_rows.append(row)

    return stat_rows


def collect_gnnexplainer_stats(smiles_list, models, data_list,
                               true_labels, pred_scores_dict, pred_labels_dict,
                               task_type="classification", expl_epochs=200):
    stat_rows = []

    for model_name, model in models.items():
        dev = next(model.parameters()).device

        scores = pred_scores_dict[model_name]
        preds = pred_labels_dict.get(model_name, None)

        model_cfg = dict(
            mode='multiclass_classification' if task_type == 'classification' else 'regression',
            task_level='graph',
            return_type='log_probs' if task_type == 'classification' else 'raw',
        )

        explainer = Explainer(
            model=WrappedModel(model),
            algorithm=GNNExplainer(epochs=expl_epochs),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=model_cfg,
        )

        for idx, (smi, data) in enumerate(zip(smiles_list, data_list)):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            d = copy.deepcopy(data)
            d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
            d = d.to(dev)

            try:
                explanation = explainer(
                    x=d.x,
                    edge_index=d.edge_index,
                    batch=d.batch,
                )
            except Exception as e:
                print(f"[{model_name}] idx={idx} GNNExplainer 失败：{e}")
                continue

            node_mask = explanation.get("node_mask")
            if node_mask is None:
                continue

            score = node_mask.cpu().numpy().flatten()
            n_atoms = len(score)
            if mol.GetNumAtoms() != n_atoms:
                continue

            row = {
                "Smiles":     smi,
                "MolIndex":   idx,
                "Model":      model_name,
                "Method":     "gnnexplainer",
                "NumAtoms":   int(n_atoms),
                "AtomMean":   float(score.mean()),
                "AtomVar":    float(score.var()),
                "AtomStd":    float(score.std()),
                "TrueLabel":  int(true_labels[idx]),
                "PredScore":  float(scores[idx]),
            }
            if task_type == "classification" and preds is not None:
                row["PredLabel"] = int(preds[idx])
            else:
                row["PredLabel"] = np.nan

            stat_rows.append(row)

    return stat_rows


# ============================
#  main
# ============================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_pt", type=str, default="dude_test.pt",
                    help="测试集 .pt 文件路径（包含 data_list 和 x_smiles/smiles）")
    ap.add_argument("--hidden_dim1", type=int, default=128)
    ap.add_argument("--hidden_dim2", type=int, default=256)
    ap.add_argument("--dropout_rate", type=float, default=0.5)
    ap.add_argument("--task_type", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--expl_epochs", type=int, default=50,
                    help="GNNExplainer 训练轮数")
    ap.add_argument("--gin_weight", type=str, default="best_GIN_model.pth")
    ap.add_argument("--gat_weight", type=str, default="best_GAT_model.pth")
    ap.add_argument("--num_per_label", type=int, default=5,
                    help="每个真实 label 抽取的分子个数")
    ap.add_argument("--output_prefix", type=str, default="pt_explain",
                    help="输出文件前缀（会存到 out/ 目录下）")
    args = ap.parse_args()

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    # 1. 读取测试集 .pt
    all_data_list, all_smiles, all_labels = load_test_pt(args.test_pt)
    print("测试集 label 分布：", {
        int(c): int((all_labels == c).sum()) for c in np.unique(all_labels)
    })

    # 2. 每个 label 抽 num_per_label 个
    selected_idx = select_indices_by_label(all_labels, num_per_label=args.num_per_label)
    print("选中样本 index：", selected_idx)

    data_list = [all_data_list[i] for i in selected_idx]
    smiles = [all_smiles[i] for i in selected_idx]
    true_labels = all_labels[selected_idx]

    sample_list_path = os.path.join(out_dir, f"{args.output_prefix}_sample_list.csv")
    pd.DataFrame({
        "Index": selected_idx,
        "Smiles": smiles,
        "TrueLabel": true_labels
    }).to_csv(sample_list_path, index=False)
    print(f"抽样分子列表写入: {sample_list_path}")

    in_dim = data_list[0].num_node_features
    out_dim = 2 if args.task_type == "classification" else 1

    # 3. 加载 GIN / GAT
    models = {
        "GIN": load_model(
            GIN, args.gin_weight,
            in_dim, args.hidden_dim1, args.hidden_dim2,
            out_dim, args.dropout_rate,
        ),
        "GAT": load_model(
            GAT, args.gat_weight,
            in_dim, args.hidden_dim1, args.hidden_dim2,
            out_dim, args.dropout_rate,
        ),
    }

    # 4. 对抽样分子做预测（PredScore / PredLabel）
    pred_scores_dict = {}
    pred_labels_dict = {}

    for name, model in models.items():
        sc, lb = predict(model, data_list, args.task_type)
        pred_scores_dict[name] = sc
        if lb is not None:
            pred_labels_dict[name] = lb

    # 5. 画图（Attention + GNNExplainer），只对抽样分子
    save_attention_images(smiles, models, data_list,
                          base_dir="images_attention_test")
    save_gnnexplainer_images(smiles, models, data_list,
                             task_type=args.task_type,
                             base_dir="images_explainer_test",
                             expl_epochs=args.expl_epochs)

    # 6. 统计表（每个分子一行：AtomMean / AtomVar / AtomStd）
    attn_stat_rows = collect_attention_stats(
        smiles, models, data_list,
        true_labels=true_labels,
        pred_scores_dict=pred_scores_dict,
        pred_labels_dict=pred_labels_dict,
        task_type=args.task_type,
    )
    expl_stat_rows = collect_gnnexplainer_stats(
        smiles, models, data_list,
        true_labels=true_labels,
        pred_scores_dict=pred_scores_dict,
        pred_labels_dict=pred_labels_dict,
        task_type=args.task_type,
        expl_epochs=args.expl_epochs,
    )

    stats_df = pd.DataFrame(attn_stat_rows + expl_stat_rows)
    stats_csv = os.path.join(out_dir, f"{args.output_prefix}_atom_stats.csv")
#    stats_df.to_csv(stats_csv, index=False)
   # print(f"抽样分子的 atom-level 统计写入: {stats_csv}")
#
    # 7. 每个分子一行的 summary（方便选 case 画图）
    mol_summary = (
        stats_df
        .groupby(["Model", "Method", "MolIndex", "Smiles", "TrueLabel"])[
            ["AtomMean", "AtomVar", "AtomStd", "PredScore", "PredLabel"]
        ]
        .first()
        .reset_index()
    )
    mol_summary_csv = os.path.join(out_dir, f"{args.output_prefix}_mol_summary.csv")
    mol_summary.to_csv(mol_summary_csv, index=False)
    print(f"per-molecule 统计写入: {mol_summary_csv}")

    # 8. 按 TrueLabel 分组，做整体汇总（Model × Method × TrueLabel）
    if args.task_type == "classification":
        class_summary = (
            stats_df
            .groupby(["Model", "Method", "TrueLabel"])[["AtomMean", "AtomVar", "AtomStd"]]
            .mean()
            .reset_index()
        )
        summary_csv = os.path.join(out_dir, f"{args.output_prefix}_class_summary.csv")
        class_summary.to_csv(summary_csv, index=False)
        print(f"按 TrueLabel 汇总的 AtomMean/AtomVar/AtomStd 写入: {summary_csv}")
        print("按类别汇总预览：")
        print(class_summary)

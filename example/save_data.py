#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import GetAdjacencyMatrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


# ===========================
#  特征函数：直接复用你现有的
# ===========================

def one_hot_encoding(value, choices):
    return [1 if choice == value else 0 for choice in choices]


def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Unknown']
    if hydrogens_implicit is False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(
        str(atom.GetHybridization()),
        ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
    )
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    vdw_radius_scaled = [
        float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)
    ]
    covalent_radius_scaled = [
        float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)
    ]

    atom_feature_vector = (
        atom_type_enc
        + n_heavy_neighbors_enc
        + formal_charge_enc
        + hybridisation_type_enc
        + is_in_a_ring_enc
        + is_aromatic_enc
        + vdw_radius_scaled
        + covalent_radius_scaled
    )

    if use_chirality:
        chirality_type_enc = one_hot_encoding(
            str(atom.GetChiralTag()),
            ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]
        )
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(
            int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]
        )
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry=True):
    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(
            str(bond.GetStereo()),
            ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"],
        )
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    data_list = []

    # 用一个小分子预先确定 feature 维度
    unrelated_mol = Chem.MolFromSmiles("O=O")
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(
        get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1))
    )

    for (smiles, y_val) in zip(x_smiles, y):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()

        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
        X = torch.tensor(X, dtype=torch.float)

        rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([torch_rows, torch_cols], dim=0)

        EF = np.zeros((n_edges, n_edge_features))
        for k, (i, j) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
        edge_attr = torch.tensor(EF, dtype=torch.float)

        y_tensor = torch.tensor([y_val], dtype=torch.float)

        data_list.append(
            Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        )

    return data_list


# ===========================
#  DUDE .ism 读取 & 划分
# ===========================

def is_valid_smiles(s):
    try:
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False


def load_ism(path, label):
    """
    加载 DUDE 目录下的 .ism 文件：
    - 自动识别任意数量的列
    - 第一列永远是 SMILES
    - 后面忽略
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",          # 任意数量的空白字符
        header=None,
        engine="python"      # python 引擎容忍度更高
    )

    # 第一列为 SMILES
    df = df.rename(columns={0: "Smiles"})
    df["Label"] = label

    # 只保留两列
    return df[["Smiles", "Label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actives", default="actives_final.ism",
                    help="活性分子 .ism 文件（label=0）")
    ap.add_argument("--decoys", default="decoys_final.ism",
                    help="decoys .ism 文件（label=1）")
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="测试集比例")
    ap.add_argument("--output_prefix", default="dude",
                    help="输出前缀：会生成 prefix_train/test.*")
    args = ap.parse_args()

    # 1) 读入 .ism 并打上标签
    df_act = load_ism(args.actives, label=0)   # 你指定 actives = 0
    df_dec = load_ism(args.decoys, label=1)   # decoys = 1

    df = pd.concat([df_act, df_dec], ignore_index=True)
    df["Smiles"] = df["Smiles"].astype(str).str.strip()

    # 2) 过滤非法 SMILES
    valid_mask = df["Smiles"].apply(is_valid_smiles)
    invalid_df = df[~valid_mask]
    if not invalid_df.empty:
        print("⚠️ 检测到非法 SMILES，将被丢弃：")
        print(invalid_df.head())

    df = df[valid_mask].reset_index(drop=True)

    # 显式展示整体分布
    print("=== 全数据集标签分布 ===")
    print(df["Label"].value_counts())

    # 3) 按 label 分层划分 train/test
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["Label"],
        random_state=42,
    )

    print("\n=== 训练集大小 ===")
    print(train_df["Label"].value_counts())
    print("\n=== 测试集大小 ===")
    print(test_df["Label"].value_counts())

    # 4) 显式保存 train/test CSV，方便你肉眼查看
    train_csv = f"{args.output_prefix}_train.csv"
    test_csv = f"{args.output_prefix}_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"\n✅ 已保存 CSV：{train_csv}, {test_csv}")

    # 5) 转成 PyG 的 DataList 并各自存 .pt
    train_smiles = train_df["Smiles"].tolist()
    train_labels = train_df["Label"].tolist()
    test_smiles = test_df["Smiles"].tolist()
    test_labels = test_df["Label"].tolist()

    train_data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
        train_smiles, train_labels
    )
    test_data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
        test_smiles, test_labels
    )

    train_pt = f"{args.output_prefix}_train.pt"
    test_pt = f"{args.output_prefix}_test.pt"

    torch.save(
        {"data_list": train_data_list,
         "smiles": train_smiles,
         "labels": train_labels},
        train_pt,
    )

    torch.save(
        {"data_list": test_data_list,
         "smiles": test_smiles,
         "labels": test_labels},
        test_pt,
    )

    print(f"✅ 已保存 PyG 数据集：{train_pt}, {test_pt}")


if __name__ == "__main__":
    main()

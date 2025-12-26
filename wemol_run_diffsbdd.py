#!/opt/conda/envs/diffsbdd4/bin/python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolToSmiles
from Bio.PDB import PDBParser, PDBIO, is_aa
import numpy as np

CWD = Path.cwd()
ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
WATERS = {"HOH", "WAT", "TIP3"}
MIN_CARBON_COUNT = 3


def run_command(cmd):
    print("▶ Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def get_min_atom_distance_and_indices(m1, m2):
    conf1, conf2 = m1.GetConformer(), m2.GetConformer()
    d_min, i1_min, i2_min = float("inf"), -1, -1
    for a1 in m1.GetAtoms():
        p1 = np.array(conf1.GetAtomPosition(a1.GetIdx()))
        for a2 in m2.GetAtoms():
            p2 = np.array(conf2.GetAtomPosition(a2.GetIdx()))
            d = np.linalg.norm(p1 - p2)
            if d < d_min:
                d_min, i1_min, i2_min = d, a1.GetIdx(), a2.GetIdx()
    return d_min, i1_min, i2_min


def estimate_min_carbons_from_sdf(sdf_path):
    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) != 2:
        raise ValueError(f"Expected 2 fragments in SDF, but found {len(frags)}")
    m1, m2 = frags
    if not m1.GetNumConformers():
        AllChem.EmbedMolecule(m1, randomSeed=42)
    if not m2.GetNumConformers():
        AllChem.EmbedMolecule(m2, randomSeed=42)
    dist, idx1, idx2 = get_min_atom_distance_and_indices(m1, m2)
    n_carbon = max(1, round(dist / 1.5))
    print(f"Estimated connecting carbon count: {n_carbon}")
    return n_carbon


def remove_hydrogens_from_sdf(sdf_path, output_path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    writer = Chem.SDWriter(str(output_path))
    for mol in supplier:
        if mol is not None:
            mol_no_h = Chem.RemoveHs(mol)
            writer.write(mol_no_h)
    writer.close()


def process_optimize_csv(sdf_out_path):
    csv_path = Path(sdf_out_path).with_suffix('.csv')
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df = df[['score', 'smiles']]
            df.columns = ['Score', 'SMILES']
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"CSV processing failed: {e}")


def split_complex(complex_pdb, protein_out, ligand_pdb_out, ligand_sdf_out, min_atoms=5, max_atoms=200):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("complex", str(complex_pdb))
    ligand_structure = struct.copy()
    for model in ligand_structure:
        for chain in list(model):
            model.detach_child(chain.id)

    ligand_count = 0
    for model in struct:
        for chain in model:
            new_chain = chain.copy()
            for res in list(new_chain):
                new_chain.detach_child(res.id)
            has_lig = False
            for res in chain:
                if is_aa(res) or res.resname.strip() in WATERS:
                    continue
                if not (min_atoms <= len(res) <= max_atoms):
                    continue
                carbon_count = sum(1 for atom in res.get_atoms() if atom.element == "C")
                if carbon_count < MIN_CARBON_COUNT:
                    continue
                new_chain.add(res.copy())
                has_lig, ligand_count = True, ligand_count + 1
            if has_lig:
                ligand_structure[0].add(new_chain)
    if ligand_count == 0:
        raise RuntimeError("No suitable ligand residues found")
    io = PDBIO()
    io.set_structure(ligand_structure)
    io.save(str(ligand_pdb_out))
    mol = Chem.MolFromPDBFile(str(ligand_pdb_out), sanitize=True, removeHs=False)
    if not mol:
        raise RuntimeError("RDKit failed to read ligand PDB")
    w = Chem.SDWriter(str(ligand_sdf_out))
    w.write(mol)
    w.close()

    protein_structure = struct.copy()
    for model in protein_structure:
        for chain in list(model):
            for res in list(chain):
                if not (is_aa(res) or res.resname.strip() in WATERS):
                    chain.detach_child(res.id)
    io.set_structure(protein_structure)
    io.save(str(protein_out))


def merge_and_deduplicate_sdfs(sdf_list, output_path):
    seen = set()
    writer = Chem.SDWriter(str(output_path))
    mol_idx = 1

    for sdf in sdf_list:
        for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
            if mol is None:
                continue

            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(frags) > 1:
                biggest = max(frags, key=lambda m: m.GetNumAtoms())
                mol = biggest
                print(f"⚠️ Found {len(frags)} fragments, keeping largest one ({mol.GetNumAtoms()} atoms).")

            try:
                smiles = MolToSmiles(mol)
            except:
                print("⚠️ SMILES generation failed, skipping molecule.")
                continue

            if smiles not in seen:
                seen.add(smiles)
                mol.SetProp("_Name", f"Mol_{mol_idx}")
                writer.write(mol)
                mol_idx += 1

    writer.close()
    print(f"🧬 Final cleaned, merged and deduplicated molecules saved to: {output_path}")

def merge_optimize_csvs(generated_sdf_files, output_csv_path, objective="sa"):

    records = []
    for sdf_file in generated_sdf_files:
        csv_path = Path(sdf_file).with_suffix(".csv")
        if not csv_path.exists():
            print(f"⚠️ CSV not found for {sdf_file}, skip merging this one.")
            continue
        try:
            df = pd.read_csv(csv_path)
            # 容错：统一列名
            if set(df.columns) >= {"score", "smiles"}:
                df = df.rename(columns={"score": "Score", "smiles": "SMILES"})
            elif set(df.columns) >= {"Score", "SMILES"}:
                pass
            else:
                print(f"⚠️ Unexpected CSV columns in {csv_path}: {list(df.columns)}; skipping.")
                continue

            # 记录批次信息（可选）
            df["BatchFile"] = Path(sdf_file).name
            records.append(df[["Score", "SMILES", "BatchFile"]])
        except Exception as e:
            print(f"⚠️ Failed to read {csv_path}: {e}")

    if not records:
        print("⚠️ No CSV files to merge.")
        return

    big = pd.concat(records, axis=0, ignore_index=True)

    # 去重策略：按 SMILES 分组，保留更优分数
    if objective.lower() == "qed":
        # 分数越大越好
        big = big.sort_values(["SMILES", "Score"], ascending=[True, False])
    else:
        # SA 越小越好（或其他默认越小越好）
        big = big.sort_values(["SMILES", "Score"], ascending=[True, True])

    big = big.drop_duplicates(subset=["SMILES"], keep="first")

    # 最终排序：便于人工查看
    if objective.lower() == "qed":
        big = big.sort_values("Score", ascending=False)
    else:
        big = big.sort_values("Score", ascending=True)

    big.to_csv(output_csv_path, index=False)
    print(f"🧾 Merged CSV saved to: {output_csv_path}")
def _canonical_smiles(smiles: str) -> str:
    try:
        m = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(m, canonical=True) if m else smiles
    except:
        return smiles

def _build_smiles_to_mol_map(sdf_files):
    mapping = {}
    for sdf in sdf_files:
        supp = Chem.SDMolSupplier(str(sdf), removeHs=False)
        for mol in supp:
            if mol is None:
                continue
            try:
                smi = _canonical_smiles(Chem.MolToSmiles(mol))
            except:
                continue
            # 只保留第一个出现的版本即可（避免重复覆盖）
            if smi not in mapping:
                mapping[smi] = mol
    return mapping

def select_top_n_from_merged(merged_csv_path, sdf_sources, out_csv_path, out_sdf_path, n=100, objective="qed"):
    """
    根据合并好的 CSV（列：Score, SMILES），选出 Top-N（qed 越大越好；sa 越小越好），
    写出一个精简版 CSV + 对应 SDF（从给定的 SDF 源中按 SMILES 精准取出；若缺失则 fallback 用 SMILES 现建 3D 构象）。
    """
    merged_csv_path = Path(merged_csv_path)
    if not merged_csv_path.exists():
        print(f"❌ Merged CSV not found: {merged_csv_path}")
        return

    try:
        df = pd.read_csv(merged_csv_path)
    except Exception as e:
        print(f"❌ Failed to read merged CSV: {e}")
        return

    # 统一列名
    cols = {c.lower(): c for c in df.columns}
    if "score" in cols and "smiles" in cols:
        df = df.rename(columns={cols["score"]: "Score", cols["smiles"]: "SMILES"})
    else:
        print(f"❌ Unexpected merged CSV columns: {list(df.columns)}")
        return

    # 去重（先按目标排序，再 drop_duplicates 保留最佳）
    if str(objective).lower() == "qed":
        df = df.sort_values(["SMILES", "Score"], ascending=[True, False])
        df = df.drop_duplicates(subset=["SMILES"], keep="first")
        df = df.sort_values("Score", ascending=False)
    else:
        # 默认认为分数越小越好（例如 SA）
        df = df.sort_values(["SMILES", "Score"], ascending=[True, True])
        df = df.drop_duplicates(subset=["SMILES"], keep="first")
        df = df.sort_values("Score", ascending=True)

    # 取前 n 条
    df_top = df.head(int(n)).copy()

    # 写 CSV
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_top.to_csv(out_csv_path, index=False)
    print(f"🏆 Top-{n} CSV saved to: {out_csv_path}")

    # 为了写 SDF，先构建 SMILES->Mol 的检索表
    # 优先使用合并后的总 SDF；若没有，就拼所有批次的 SDF
    sdf_sources = [Path(p) for p in sdf_sources if Path(p).exists()]
    smiles2mol = _build_smiles_to_mol_map(sdf_sources)

    writer = Chem.SDWriter(str(out_sdf_path))
    kept = 0
    for _, row in df_top.iterrows():
        smi = _canonical_smiles(row["SMILES"])
        mol = smiles2mol.get(smi, None)
        if mol is None:
            # fallback：用 SMILES 重新构建一个分子，补一个 3D 构象，保证 SDF 能写出来
            m = Chem.MolFromSmiles(smi)
            if m is None:
                print(f"⚠️ Cannot rebuild from SMILES: {smi}")
                continue
            m = Chem.AddHs(m)
            try:
                AllChem.EmbedMolecule(m, randomSeed=42)
                AllChem.UFFOptimizeMolecule(m, maxIters=200)
            except Exception as e:
                print(f"⚠️ Embed/opt failed for {smi}: {e}")
            m.SetProp("_Name", f"Top_{kept+1}")
            m.SetProp("Score", str(row["Score"]))
            writer.write(m)
            kept += 1
            continue

        # 从源 SDF 拿到的 mol，补充一下属性再写
        try:
            mol.SetProp("_Name", mol.GetProp("_Name") if mol.HasProp("_Name") else f"Top_{kept+1}")
        except:
            mol.SetProp("_Name", f"Top_{kept+1}")
        mol.SetProp("Score", str(row["Score"]))
        writer.write(mol)
        kept += 1

    writer.close()
    print(f"🧪 Top-{n} SDF saved to: {out_sdf_path} (kept {kept} molecules)")


def main():
    parser = argparse.ArgumentParser("Unified DiffSBDD Runner")
    parser.add_argument("--mode", choices=["design", "inpaint", "optimize", "inpaint_auto"], required=True)
    parser.add_argument("--pdb_complex", required=True)
    parser.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "crossdocked_fullatom_cond.ckpt"))
    parser.add_argument("--outfile")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--center", choices=["ligand", "pocket"], default="ligand")
    parser.add_argument("--add_n_nodes", type=int, default=10)
    parser.add_argument("--objective", choices=["sa", "qed"], default="sa")
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--evolution_steps", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    complex_path = Path(args.pdb_complex).resolve()
    protein_pdb = CWD / "protein_only.pdb"
    ligand_pdb = CWD / "ligand_only.pdb"
    ligand_sdf = CWD / "ligand_only.sdf"
    ligand_sdf_noH = CWD / "ligand_only_noH.sdf"

    print("🔍 Splitting complex ...")
    split_complex(complex_path, protein_pdb, ligand_pdb, ligand_sdf)
    print("🧼 Removing hydrogens ...")
    remove_hydrogens_from_sdf(ligand_sdf, ligand_sdf_noH)

    generated_files = []

    # ---------------------------
    # 批次执行器
    # ---------------------------
    def build_and_run(base_cmd, output_prefix):
        if args.mode == "optimize":
            n = args.n_samples
            batch_size = args.population_size
            chunks = (n + batch_size - 1) // batch_size
            for i in range(chunks):
                print(f"🚀 Running optimize batch {i + 1}/{chunks} ...")
                batch_seed = (args.seed or 0) + i
                out_file = Path(f"{output_prefix}_{i + 1}.sdf")
                cmd = base_cmd + ["--outfile", str(out_file), "--seed", str(batch_seed)]
                run_command(cmd)
                process_optimize_csv(out_file)  # 规范列名为 Score/SMILES
                generated_files.append(out_file)
        else:
            # design/inpaint：每批最多 50 个，并做 1.2x 冗余
            chunks = max(1, (args.n_samples + 49) // 50)
            redundant_total = int(np.ceil(args.n_samples * 1.2))
            for i in range(chunks):
                n = min(50, redundant_total - i * 50)
                if n <= 0:
                    break
                print(f"🚀 Generating batch {i + 1}/{chunks} with {n} molecules ...")
                batch_seed = (args.seed or 0) + i
                out_file = Path(f"{output_prefix}_{i + 1}.sdf")
                cmd = base_cmd + ["--n_samples", str(n), "--outfile", str(out_file), "--seed", str(batch_seed)]
                run_command(cmd)
                generated_files.append(out_file)

    # ---------------------------
    # 组装指令并运行
    # ---------------------------
    if args.mode == "design":
        base_cmd = [
            sys.executable, str(ROOT / "generate_ligands.py"),
            str(args.checkpoint),
            "--pdbfile", str(protein_pdb),
            "--ref_ligand", str(ligand_sdf_noH),
        ]
        build_and_run(base_cmd, "design")

    elif args.mode == "inpaint":
        base_cmd = [
            sys.executable, str(ROOT / "inpaint.py"),
            str(args.checkpoint),
            "--pdbfile", str(protein_pdb),
            "--ref_ligand", str(ligand_sdf_noH),
            "--fix_atoms", str(ligand_sdf_noH),
            "--add_n_nodes", str(args.add_n_nodes),
        ]
        build_and_run(base_cmd, "inpaint")

    elif args.mode == "optimize":
        base_cmd = [
            sys.executable, str(ROOT / "optimize2.py"),
            "--checkpoint", args.checkpoint,
            "--pdbfile", str(protein_pdb),
            "--ref_ligand", str(ligand_sdf_noH),
            "--objective", args.objective,
            "--population_size", str(args.population_size),
            "--evolution_steps", str(args.evolution_steps),
            "--top_k", str(args.top_k),
            "--timesteps", str(args.timesteps),
        ]
        build_and_run(base_cmd, "optimize")

    elif args.mode == "inpaint_auto":
        min_carbons = estimate_min_carbons_from_sdf(ligand_sdf)
        total_samples = args.n_samples
        max_per_batch = 50
        redundant_total = int(np.ceil(total_samples * 1.2))
        total_batches = max(1, (redundant_total + max_per_batch - 1) // max_per_batch)

        for j, add_nodes in enumerate(range(min_carbons + 4, min_carbons + 10), 1):
            print(f"🔁 add_n_nodes = {add_nodes}, generating {total_samples} molecules in {total_batches} batches ...")
            for i in range(total_batches):
                n = min(max_per_batch, redundant_total - i * max_per_batch)
                if n <= 0:
                    break
                out = CWD / f"inpaint_auto_{j}_{i + 1}.sdf"
                batch_seed = (args.seed or 0) + j * 100 + i
                cmd = [
                    sys.executable, str(ROOT / "inpaint.py"),
                    str(args.checkpoint),
                    "--pdbfile", str(protein_pdb),
                    "--outfile", str(out),
                    "--ref_ligand", str(ligand_sdf_noH),
                    "--fix_atoms", str(ligand_sdf_noH),
                    "--add_n_nodes", str(add_nodes),
                    "--n_samples", str(n),
                    "--seed", str(batch_seed),
                ]
                print(f"🚀 Generating with add_n_nodes={add_nodes}, batch {i + 1}/{total_batches}, n={n}")
                run_command(cmd)
                generated_files.append(out)

    # ---------------------------
    # 合并 & 选 Top-N
    # ---------------------------
    if generated_files:
        # 非 optimize：仅做 SDF 合并/去重（若提供 --outfile），不做 CSV/Top-N
        if args.mode != "optimize":
            if args.outfile:
                print(f"📦 (non-optimize) Merging and deduplicating {len(generated_files)} SDFs into: {args.outfile}")
                merge_and_deduplicate_sdfs(generated_files, args.outfile)
            else:
                print("ℹ️ (non-optimize) No --outfile provided; keep batch SDFs as-is. Skipping CSV merge and Top-N.")
            return

        # ========= 以下仅 optimize =========
        # 合并 SDF（可选）并确定 CSV/SDF 来源
        if args.outfile:
            print(f"📦 Merging and deduplicating {len(generated_files)} files (SDF) ...")
            merge_and_deduplicate_sdfs(generated_files, args.outfile)
            merged_csv_path = Path(args.outfile).with_suffix(".csv")
            sdf_sources = [args.outfile]  # Top-N 优先从合并后的 SDF 取
        else:
            merged_csv_path = CWD / "optimize_merged.csv"
            sdf_sources = generated_files

        # 合并批次 CSV
        print("📊 Merging batch CSV files into one ...")
        merge_optimize_csvs(generated_files, merged_csv_path, objective=args.objective)

        # 基于合并 CSV 选 Top-N（若前一步没产出 CSV，会在函数里优雅退出）
        top_n = int(args.n_samples)
        if args.outfile:
            top_csv_out = Path(args.outfile).with_name(f"{Path(args.outfile).stem}_top{top_n}.csv")
            top_sdf_out = Path(args.outfile).with_name(f"{Path(args.outfile).stem}_top{top_n}.sdf")
        else:
            stem = f"{args.mode}_top{top_n}"
            top_csv_out = CWD / f"{stem}.csv"
            top_sdf_out = CWD / f"{stem}.sdf"

        print(f"🏁 Selecting Top-{top_n} molecules by {args.objective} ...")
        select_top_n_from_merged(
            merged_csv_path=merged_csv_path,
            sdf_sources=sdf_sources,
            out_csv_path=top_csv_out,
            out_sdf_path=top_sdf_out,
            n=top_n,
            objective=args.objective
        )


if __name__ == "__main__":
    main()

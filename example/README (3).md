# DUD-E Pipeline README

📘 DUD-E 全流程项目 README（总说明文档）



本 README 用于说明本项目中所有脚本的作用、输入/输出格式、运行方式，以及整个 pipeline 的目的和产物结构。



项目由以下主要组件构成：



\*\*数据处理：\*\*将 .ism 文件转换为 GNN 可用的 PyG .pt 格式

– 使用 save\_data.py 



save\_data



\*\*模型训练：\*\*在每个靶点上训练 GIN/GAT 模型并输出 PDF 报告

– 使用 训练dude.py 



训练dude



\*\*可解释性分析：\*\*Attention 可视化与 GNNExplainer

– 使用 画图.py（解释性脚本）



画图



\*\*跨靶点统计：\*\*计算 Mean\_diff / Var\_mean / Std\_mean

– 使用 统计差异.py 



统计差异



\*\*批量自动化：\*\*对所有靶点自动运行上述脚本

– 使用 run\_batch.py 



run\_batch



\*\*目录树生成（可选）：\*\*输出目录结构

– 使用 文件结构.py（dirtree.py）



文件结构



\*\*清理压缩包（可选）：\*\*删除 .gz/.zip/.tar

– 使用 删除文件夹.py 



删除文件夹



📍 1. 项目总体目的 Purpose



本项目旨在对 DUD-E 全部靶点进行：



数据标准化（SMILES → PyG 图）



模型训练（GIN/GAT）



性能评估（Acc / AUC）



解释性可视化（Attention、GNNExplainer）



跨靶点统计分析（结构差异、模型关注模式）



自动化 pipeline（无需人工重复操作）



最终产出可用于可解释性实验部分，为模型能够区分“优质 vs 劣质分子”的论证提供统计证据。



📂 2. 目录结构（输出示例）

all/

&nbsp;├── aa2ar/

&nbsp;│    ├── actives\_final.ism

&nbsp;│    ├── decoys\_final.ism

&nbsp;│    ├── dude\_train.pt

&nbsp;│    ├── dude\_test.pt

&nbsp;│    ├── best\_GIN\_model.pth

&nbsp;│    ├── best\_GAT\_model.pth

&nbsp;│    ├── aa2ar\_gat\_gin\_report.pdf

&nbsp;│    ├── out/

&nbsp;│    │     ├── aa2ar\_explain\_sample\_list.csv

&nbsp;│    │     ├── aa2ar\_explain\_mol\_summary.csv

&nbsp;│    │     ├── aa2ar\_explain\_class\_summary.csv

&nbsp;│    │     ├── images\_attention\_test/

&nbsp;│    │     └── images\_explainer\_test/

&nbsp;├── cdk2/

&nbsp;├── ...（所有靶点）





如需自动生成树：



python dirtree.py all



🧩 3. 各脚本说明（目的 / 输入 / 输出）

3.1 save\_data.py — 生成训练/测试集



save\_data



目的



将 DUDE 的 .ism 文件读取为 SMILES



清洗非法 SMILES



按 stratify（分层）划分 train/test



转换为 PyG 的图结构 (Data)



导出：xxx\_train.pt / xxx\_test.pt + CSV 版本



输入

--actives       活性分子 ism（label=0）

--decoys        decoys ism（label=1）

--test\_size     测试集比例

--output\_prefix 输出前缀，例如 “dude”



输出



dude\_train.csv（SMILES+Label）



dude\_test.csv



dude\_train.pt



dude\_test.pt



3.2 训练dude.py — 靶点级模型训练



训练dude



目的



在单个靶点上训练两套模型：



GAT（二层）



GIN（二层）



AttentionPooling 作为图级汇聚



自动：



保存最优模型权重



生成训练曲线 + ROC 曲线 + summary 的 PDF 报告



输入

--train\_pt       dude\_train.pt

--test\_pt        dude\_test.pt

--hidden\_dim1    隐层1维度

--hidden\_dim2    隐层2维度

--epochs         训练轮数

--batch\_size     批大小

--output\_pdf     输出报告



输出



best\_GIN\_model.pth



best\_GAT\_model.pth



xxx\_gat\_gin\_report.pdf



最终 stdout 中包含：



\[GIN] Final best model on test set | Accuracy: XXX | AUC: XXX

\[GAT] Final best model on test set | Accuracy: XXX | AUC: XXX



3.3 画图.py — 可解释性可视化与统计



画图



目的



对测试集中抽样得到的分子：



生成 Attention 热图



生成 GNNExplainer 热图



计算 AtomMean / AtomVar / AtomStd



生成：



per-molecule summary



per-class summary（核心统计结果）



输入

--test\_pt           dude\_test.pt

--gin\_weight        best\_GIN\_model.pth

--gat\_weight        best\_GAT\_model.pth

--num\_per\_label     每类抽取多少分子

--expl\_epochs       GNNExplainer 训练轮数

--output\_prefix     eg: aa2ar\_explain



输出（都写入各靶点目录的 out/）

out/

&nbsp;├── aa2ar\_explain\_sample\_list.csv

&nbsp;├── aa2ar\_explain\_mol\_summary.csv

&nbsp;├── aa2ar\_explain\_class\_summary.csv

&nbsp;├── images\_attention\_test/

&nbsp;└── images\_explainer\_test/





重点统计文件用于最终论文实验。



3.4 统计差异.py — 跨靶点综合统计



统计差异



目的



汇总全部靶点的解释性指标：



Mean\_diff\_abs



Var\_mean



Std\_mean



自动生成：



barplot（均值）



boxplot（分布）



violin（分布形态）



关键分位点 CSV



输入

BASE\_DIR = all/   # 每个靶点的 out/ 内应有 class\_summary.csv



输出



写入 all\_summary/：



all\_targets\_raw.csv

all\_targets\_summary.csv

bar\_\*.png

box\_\*.png

violin\_\*.png

boxplot\_key\_stats.csv



3.5 run\_batch.py — 端到端自动化运行所有靶点



run\_batch



目的



一次性自动运行以下步骤：



save\_data.py



训练dude.py



画图.py



汇总所有靶点的训练指标（Acc/AUC）



适合跑完整个 DUD-E。



输入

--all\_dir             存放所有靶点的目录

--save\_data\_script    save\_data.py 路径

--train\_script        训练脚本路径

--explain\_script      画图脚本路径

--test\_size           测试集比例

--num\_per\_label       抽样数量



输出



每个靶点自动生成 train/test/model/解释结果



全局 dude\_targets\_summary.csv



3.6 文件结构.py（dirtree.py）— 打印目录树结构



文件结构



用于展示项目目录结构，论文写法或展示使用。


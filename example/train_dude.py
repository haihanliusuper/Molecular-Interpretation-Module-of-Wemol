#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

from torch.nn import Sequential as Seq, Linear, ReLU, Dropout
from torch.utils.data import WeightedRandomSampler

from torch_geometric.nn import GATConv, GINConv
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ===========================
#   命令行参数
# ===========================
parser = argparse.ArgumentParser(description="GAT & GIN Training on pre-split train/test .pt")
parser.add_argument("--train_pt", type=str, default="dude_train.pt",
                    help="Path to the train .pt file")
parser.add_argument("--test_pt", type=str, default="dude_test.pt",
                    help="Path to the test .pt file")
parser.add_argument("--hidden_dim1", type=int, default=128,
                    help="Dimension of the first hidden layer")
parser.add_argument("--hidden_dim2", type=int, default=256,
                    help="Dimension of the second hidden layer")
parser.add_argument("--dropout_rate", type=float, default=0.5,
                    help="Dropout rate")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--output_pdf", type=str, default="gat_gin_report.pdf",
                    help="Output PDF path")
args = parser.parse_args()

train_pt_path = args.train_pt
test_pt_path = args.test_pt
hidden_dim1 = args.hidden_dim1
hidden_dim2 = args.hidden_dim2
dropout_rate = args.dropout_rate
num_epochs = args.epochs
batch_size = args.batch_size
output_pdf_path = args.output_pdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
#   注意力池化模块
# ===========================
class AttentionPooling(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.attention_mechanism = nn.Linear(node_dim, 1)
        self.mask_layer = nn.Linear(node_dim, 1)

    def forward(self, node_feats, batch_idx):
        # node_feats: [N, D]
        attn_scores = self.attention_mechanism(node_feats)   # [N, 1]
        mask_logits = self.mask_layer(node_feats)            # [N, 1]
        node_mask = torch.sigmoid(mask_logits)               # [N, 1]

        final_scores = attn_scores * node_mask               # [N, 1]
        pooled = scatter(node_feats * final_scores, batch_idx,
                         dim=0, reduce="sum")
        return pooled


# ===========================
#   GAT 模型
# ===========================
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,
                 output_dim, dropout_rate=0.5, heads=8):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim1, heads=heads)
        self.conv2 = GATConv(hidden_dim1 * heads, hidden_dim2, heads=1)
        self.dropout = Dropout(dropout_rate)
        self.out = Linear(hidden_dim2, output_dim)
        self.attention_pooling = AttentionPooling(hidden_dim2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.attention_pooling(x, batch)
        x = self.out(x)
        return x


# ===========================
#   GIN 模型
# ===========================
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,
                 output_dim, dropout_rate=0.5):
        super().__init__()
        self.mlp1 = Seq(
            Linear(input_dim, hidden_dim1),
            ReLU(),
            Linear(hidden_dim1, hidden_dim1),
        )
        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = Seq(
            Linear(hidden_dim1, hidden_dim2),
            ReLU(),
            Linear(hidden_dim2, hidden_dim2),
        )
        self.conv2 = GINConv(self.mlp2)

        self.dropout = Dropout(dropout_rate)
        self.out = Linear(hidden_dim2, output_dim)
        self.attention_pooling = AttentionPooling(hidden_dim2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.attention_pooling(x, batch)
        x = self.out(x)
        return x


# ===========================
#   加载 train/test 数据
# ===========================
def load_pt_data(path):
    obj = torch.load(path)
    data_list = obj["data_list"]
    return data_list

train_data_list = load_pt_data(train_pt_path)
test_data_list = load_pt_data(test_pt_path)

num_features = train_data_list[0].num_node_features
print(f"Number of node features: {num_features}")

# 假设是二分类
num_classes = 2

# ===========================
#   类别平衡：class_weight + WeightedRandomSampler
# ===========================
train_labels = np.array([int(d.y.item()) for d in train_data_list])
unique_classes = np.unique(train_labels)
print("Train label distribution:", {int(c): int((train_labels == c).sum()) for c in unique_classes})

# 1) 损失函数里的类别权重
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=train_labels,
)
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 2) 采样层面的类别平衡（每个样本一个采样权重）
sample_weights = np.zeros_like(train_labels, dtype=np.float32)
for idx, c in enumerate(unique_classes):
    sample_weights[train_labels == c] = class_weights[idx].item()

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights),
    num_samples=len(sample_weights),
    replacement=True,
)

# DataLoader：训练用 sampler（类别均衡采样），测试正常顺序
train_loader = DataLoader(train_data_list, batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)


# ===========================
#   训练 & 测试函数
# ===========================
def train_one_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    predictions = []
    true_values = []
    losses = []
    y_scores = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.long())
            total_loss += loss.item()

            batch_loss = F.cross_entropy(output, data.y.long(), reduction="none")
            losses.extend(batch_loss.cpu().numpy())

            preds = output.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_values.extend(data.y.cpu().numpy())

            if output.shape[1] == 2:
                y_score = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            else:
                y_score = F.softmax(output, dim=1).cpu().numpy()
            y_scores.extend(y_score)

    accuracy = accuracy_score(true_values, predictions)
    cm = confusion_matrix(true_values, predictions)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    class_loss = {}
    unique_classes_eval = np.unique(true_values)
    for c in unique_classes_eval:
        mask = np.array(true_values) == c
        class_loss[c] = np.mean(np.array(losses)[mask]) if np.sum(mask) > 0 else 0.0

    roc_auc = None
    if len(unique_classes_eval) == 2:
        fpr, tpr, _ = roc_curve(true_values, y_scores)
        roc_auc = auc(fpr, tpr)

    return {
        "total_loss": total_loss / len(loader),
        "accuracy": accuracy,
        "class_accuracies": class_accuracies,
        "class_losses": class_loss,
        "roc_auc": roc_auc,
        "y_scores": y_scores,
        "true_values": true_values,
        "predictions": predictions,
    }


# ===========================
#   初始化模型 & 训练
# ===========================
models = {
    "GAT": GAT(
        input_dim=num_features,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        output_dim=num_classes,
        dropout_rate=dropout_rate,
    ),
    "GIN": GIN(
        input_dim=num_features,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        output_dim=num_classes,
        dropout_rate=dropout_rate,
    ),
}

optimizers = {
    name: torch.optim.Adam(model.parameters(), lr=0.0001)
    for name, model in models.items()
}

all_models_results = {}
all_histories = {}

for model_name, model in models.items():
    model = model.to(device)
    optimizer = optimizers[model_name]

    train_losses = []
    test_losses = []
    best_accuracy = 0.0

    print(f"\n==== Training {model_name} ====")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader)
        test_results = evaluate(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_results["total_loss"])

        if test_results["accuracy"] > best_accuracy:
            best_accuracy = test_results["accuracy"]
            torch.save(model.state_dict(), f"best_{model_name}_model.pth")

        print(
            f"[{model_name}] Epoch {epoch+1}/{num_epochs} "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_results['total_loss']:.4f} | "
            f"Accuracy: {test_results['accuracy']:.4f}"
        )

    # 加载最佳模型，做最终评估
    model.load_state_dict(torch.load(f"best_{model_name}_model.pth"))
    final_results = evaluate(model, test_loader)

    all_models_results[model_name] = final_results
    all_histories[model_name] = {
        "train_losses": train_losses,
        "test_losses": test_losses,
    }
    roc_auc = final_results["roc_auc"]
    auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    print(
        f"[{model_name}] Final best model on test set | "
        f"Accuracy: {final_results['accuracy']:.4f} | "
        f"AUC: {auc_str}"
    )


# ===========================
#   画图写 PDF
# ===========================
with PdfPages(output_pdf_path) as pdf:
    for model_name, results in all_models_results.items():
        history = all_histories[model_name]
        train_losses = history["train_losses"]
        test_losses = history["test_losses"]

        # 1) 训练/测试 loss 曲线
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name} Training and Testing Loss")
        plt.legend()
        pdf.savefig()
        plt.close()

        # 2) ROC 曲线（二分类）
        if results["roc_auc"] is not None:
            fpr, tpr, _ = roc_curve(results["true_values"], results["y_scores"])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            pdf.savefig()
            plt.close()

        # 3) 文本总结页
        summary_text = f"{model_name} Best Model Accuracy: {results['accuracy']:.4f}\n"
        for i, acc in enumerate(results["class_accuracies"]):
            summary_text += f"Class {i} Accuracy: {acc:.4f}\n"

        plt.figure(figsize=(8, 2))
        plt.text(0.01, 0.5, summary_text, fontsize=12, va="center")
        plt.axis("off")
        pdf.savefig()
        plt.close()

print(f"Training complete. All plots saved to '{output_pdf_path}'.")

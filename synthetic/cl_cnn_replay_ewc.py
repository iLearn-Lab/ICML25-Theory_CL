import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import scienceplots
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

plt.style.use('science')


# =========================================================
# 1. 网络结构：LinearCNN + EWC 模块
# =========================================================
class LinearCNN(nn.Module):
    """1-D CNN with two-class head, plus built-in EWC support."""

    def __init__(self, input_dim: int, out_channel: int,
                 patch_num: int, lamda: float = 1000.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_channel * 2,
            kernel_size=input_dim // patch_num,
            stride=input_dim // patch_num,
        )
        self.out_channel = out_channel
        self.patch_num = patch_num
        self.patch_size = input_dim // patch_num
        self.lamda = lamda  # EWC strength

    # ---------- 特征提取 ----------
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.pow(x, 3)
        x = torch.mean(x, dim=2)  # (N, 2c)
        return x  # latent representation

    # ---------- logits ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature(x)  # (N, 2c)
        out = torch.stack(
            [
                torch.sum(feats[:, : self.out_channel], dim=1),
                torch.sum(feats[:, self.out_channel :], dim=1),
            ],
            dim=1,
        )  # (N,2)
        return out

    # ---------- EWC 工具函数 ----------
    def _is_on_cuda(self) -> bool:
        return next(self.parameters()).is_cuda

    # ---------- EWC 工具函数（完全版 Fisher）----------
    def estimate_fisher(self, dataset, sample_size: int, batch_size: int = 32):
        """用 dataset 随机采样 sample_size 个样本估计 Fisher 对角线."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loglikelihoods = []

        for x, y in loader:
            if self._is_on_cuda():
                x, y = x.cuda(), y.cuda()
            logp = F.log_softmax(self(x), dim=1)
            ll = logp[range(x.size(0)), y]
            loglikelihoods.append(ll)
            if len(loglikelihoods) * batch_size >= sample_size:
                break

        fisher_diagonals = None
        for ll in torch.cat(loglikelihoods):
            self.zero_grad()
            ll.backward(retain_graph=True)
            grads = [p.grad.clone() for p in self.parameters()]
            if fisher_diagonals is None:
                fisher_diagonals = [g.pow(2) for g in grads]
            else:
                fisher_diagonals = [
                    f + g.pow(2) for f, g in zip(fisher_diagonals, grads)
                ]

        fisher_diagonals = [f / len(loglikelihoods) for f in fisher_diagonals]
        names = [n.replace(".", "__") for n, _ in self.named_parameters()]
        return {n: f.detach() for n, f in zip(names, fisher_diagonals)}

    def consolidate(self, fisher_dict):
        """保存 θ̂ 和 Fisher，对后续任务施加 EWC 惩罚."""
        for n, p in self.named_parameters():
            name = n.replace(".", "__")
            self.register_buffer(f"{name}_mean", p.data.clone())
            self.register_buffer(f"{name}_fisher", fisher_dict[name].clone())

    def ewc_loss(self):
        losses = []
        for n, p in self.named_parameters():
            name = n.replace(".", "__")
            try:
                mean = getattr(self, f"{name}_mean")
                fisher = getattr(self, f"{name}_fisher")
                losses.append((fisher * (p - mean).pow(2)).sum())
            except AttributeError:
                # 首任务之前还没有 consolidate
                continue
        if losses:
            return (self.lamda / 2) * sum(losses)
        return torch.tensor(0.0, device=p.device)


# =========================================================
# 2. 训练 / 评估工具
# =========================================================
def test(model, criterion, data, labels):
    with torch.no_grad():
        outputs = model(data)
        loss = criterion(outputs, labels)
    return loss.item()


def test_ACC(model, data, labels):
    with torch.no_grad():
        outputs = model(data)
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        # 返回 error_rate，保持与原脚本一致
        return 1 - correct / data.size(0)


def train_with_Margin(
    model,
    criterion,
    data,
    labels,
    optimizer,
    test_data,
    test_labels,
    epochs,
):
    """Baseline training loop (no distillation)."""

    Largest, Smallest, trainR, testR, testtR = [], [], [], [], []
    for epoch in range(epochs):
        funcf = model(data)[:, 1] - model(data)[:, 0]
        yfuncf = funcf * (labels - 0.5) * 2  # Convert labels to {-1, 1}
        Largest.append(yfuncf.max().detach().item())
        Smallest.append(yfuncf.min().detach().item())
        testR.append(test_ACC(model, test_data, test_labels))
        testtR.append(test(model, criterion, test_data, test_labels))
        trainR.append(test(model, criterion, data, labels))

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return Largest, Smallest, testR, testtR, trainR


def train_with_Margin_ewc(model, criterion, data, labels, optimizer,
                          test_data, test_labels, epochs):
    """交叉熵 + EWC"""
    Largest, Smallest, trainR, testR, testtR = [], [], [], [], []
    for _ in range(epochs):
        funcf = model(data)[:, 1] - model(data)[:, 0]
        yfuncf = funcf * (labels - 0.5) * 2
        Largest.append(yfuncf.max().item())
        Smallest.append(yfuncf.min().item())
        testR.append(test_ACC(model, test_data, test_labels))
        testtR.append(test(model, criterion, test_data, test_labels))
        trainR.append(test(model, criterion, data, labels))

        optimizer.zero_grad()
        cls_loss = criterion(model(data), labels)
        loss = cls_loss + model.ewc_loss()
        loss.backward()
        optimizer.step()
    return Largest, Smallest, testR, testtR, trainR


# =========================================================
# Main program – sweep rotation angle 1° … 180°
# =========================================================
if __name__ == "__main__":
    # --------------------- reproducibility ---------------------
    random.seed(205348320)
    torch.manual_seed(205348320)
    np.random.seed(205348320)

    # --------------------- hyper-parameters --------------------
    num_epochs = 3000  # ← may reduce for faster runs
    DATA_NUM_TASK1 = 100
    TEST_DATA_NUM_TASK1 = 1000
    DATA_NUM_TASK2 = 100
    TEST_DATA_NUM_TASK2 = 1000

    CLUSTER_NUM = 1
    PATCH_NUM = 2
    PATCH_LEN = 100
    Noiselevel = 1.0
    bmu = 8

    input_dim = 2 * PATCH_LEN
    out_channel = 10  # m value

    # --------------------- data generation utils ---------------
    def generate_task_data(data_num, test_num, feature_vec):
        data, labels = [], []
        for _ in range(data_num + test_num):
            y = np.random.choice([-1, 1])
            xi = torch.tensor(
                np.random.normal(0, Noiselevel, size=(PATCH_LEN)),
                dtype=torch.float32,
            )
            x = torch.stack([feature_vec * bmu * y, xi])
            idx = torch.randperm(len(x))
            x = x[idx].flatten()
            data.append(x)
            labels.append(0 if y == -1 else 1)
        return torch.stack(data), torch.tensor(labels)

    # --------------------- Task 1 -------------------------------
    base_feat = torch.zeros(CLUSTER_NUM, PATCH_LEN)
    base_feat[0, 0] = 1  # one-hot feature

    data1, labels1 = generate_task_data(
        DATA_NUM_TASK1, TEST_DATA_NUM_TASK1, base_feat[0]
    )

    training_data1 = data1[:DATA_NUM_TASK1].unsqueeze(1).float()
    test_data1 = data1[DATA_NUM_TASK1:].unsqueeze(1).float()
    training_labels1 = labels1[:DATA_NUM_TASK1].long()
    test_labels1 = labels1[DATA_NUM_TASK1:].long()

    # ----- train Task 1 -----
    model = LinearCNN(input_dim, out_channel, PATCH_NUM)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_with_Margin(
        model,
        criterion,
        training_data1,
        training_labels1,
        optimizer,
        test_data1,
        test_labels1,
        num_epochs,
    )

    # save snapshot (teacher network for replay)
    teacher_model = deepcopy(model)
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_state = deepcopy(teacher_model.state_dict())

    # --------------------- sweep over rotation angles ----------
    angles = np.arange(0, 181, 1)
    errors_no, errors_mix, errors_rep = [], [], []
    errors_mix_ewc = []  # ### NEW

    for ang in angles:
        rotation_angle = np.deg2rad(ang)
        rot = torch.tensor(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ],
            dtype=torch.float32,
        )
        b = base_feat[0].clone()
        b[:2] = torch.matmul(rot, b[:2])  # apply rotation in first two dims

        # generate Task 2 data
        data2, labels2 = generate_task_data(
            DATA_NUM_TASK2, TEST_DATA_NUM_TASK2, b
        )
        training_data2 = data2[:DATA_NUM_TASK2].unsqueeze(1).float()
        test_data2 = data2[DATA_NUM_TASK2:].unsqueeze(1).float()
        training_labels2 = labels2[:DATA_NUM_TASK2].long()
        test_labels2 = labels2[DATA_NUM_TASK2:].long()

        # ------------------------------------------------------
        # Variant A – no mixing
        # ------------------------------------------------------
        model_no = LinearCNN(input_dim, out_channel, PATCH_NUM)
        model_no.load_state_dict(teacher_state)
        opt_no = torch.optim.SGD(model_no.parameters(), lr=0.01)
        train_with_Margin(
            model_no,
            criterion,
            training_data2,
            training_labels2,
            opt_no,
            test_data2,
            test_labels2,
            num_epochs,
        )

        # ------------------------------------------------------
        # Variant B – 90/10 mixing
        # ------------------------------------------------------
        model_mix = LinearCNN(input_dim, out_channel, PATCH_NUM)
        model_mix.load_state_dict(teacher_state)
        opt_mix = torch.optim.SGD(model_mix.parameters(), lr=0.01)

        idx2 = torch.randperm(DATA_NUM_TASK2)[: int(DATA_NUM_TASK2 * 0.9)]
        idx1 = torch.randperm(DATA_NUM_TASK1)[: int(DATA_NUM_TASK1 * 0.1)]
        training_data_mix = torch.cat(
            [training_data2[idx2], training_data1[idx1]], dim=0
        )
        training_labels_mix = torch.cat(
            [training_labels2[idx2], training_labels1[idx1]], dim=0
        )
        shuf = torch.randperm(training_data_mix.size(0))
        training_data_mix = training_data_mix[shuf]
        training_labels_mix = training_labels_mix[shuf]

        train_with_Margin(
            model_mix,
            criterion,
            training_data_mix,
            training_labels_mix,
            opt_mix,
            test_data2,
            test_labels2,
            num_epochs,
        )

        # ------------------------------------------------------
        # Variant C – Exact EWC（你的原实现）
        # ------------------------------------------------------
        model_ewc = LinearCNN(input_dim, out_channel, PATCH_NUM, lamda=1000000.0)
        model_ewc.load_state_dict(teacher_state)
        # ------ Fisher 估计 ------
        task1_ds = TensorDataset(training_data1, training_labels1)
        fisher_info = model_ewc.estimate_fisher(
            task1_ds, sample_size=training_data1.size(0), batch_size=training_data1.size(0)
        )
        model_ewc.consolidate(fisher_info)

        opt_ewc = torch.optim.SGD(model_ewc.parameters(), lr=0.01)

        train_with_Margin_ewc(
            model_ewc, criterion,
            training_data2, training_labels2,
            opt_ewc,
            test_data2, test_labels2,
            num_epochs
        )

        # ------------------------------------------------------
        # Variant D – 90/10 mixing + EWC（新增）
        # ------------------------------------------------------
        model_mix_ewc = LinearCNN(input_dim, out_channel, PATCH_NUM, lamda=1000000.0)  # ### NEW
        model_mix_ewc.load_state_dict(teacher_state)  # ### NEW
        # 仍用 Task 1 数据估计 Fisher 并固化  # ### NEW
        fisher_info_mix = model_mix_ewc.estimate_fisher(  # ### NEW
            task1_ds, sample_size=training_data1.size(0), batch_size=training_data1.size(0)
        )
        model_mix_ewc.consolidate(fisher_info_mix)  # ### NEW
        opt_mix_ewc = torch.optim.SGD(model_mix_ewc.parameters(), lr=0.01)  # ### NEW

        # 在混合数据上训练，但损失函数=CE + EWC正则  # ### NEW
        train_with_Margin_ewc(  # ### NEW
            model_mix_ewc, criterion,
            training_data_mix, training_labels_mix,
            opt_mix_ewc,
            test_data2, test_labels2,
            num_epochs
        )

        # ----------------- evaluate Task 1 retention -----------
        def eval_error(model_candidate):
            with torch.no_grad():
                outputs = model_candidate(test_data1)
                acc = (
                    torch.argmax(outputs, dim=1) == test_labels1
                ).float().mean().item()
            return 1.0 - acc  # error rate

        errors_no.append(eval_error(model_no))
        errors_mix.append(eval_error(model_mix))
        errors_rep.append(eval_error(model_ewc))
        errors_mix_ewc.append(eval_error(model_mix_ewc))  # ### NEW

        print(
            f"Angle {ang:3d}°  |  Err (No)={errors_no[-1]:.3f}  "
            f"Err (Mix)={errors_mix[-1]:.3f}  Err (EWC)={errors_rep[-1]:.3f}  "
            f"Err (Mix+EWC)={errors_mix_ewc[-1]:.3f}"  # ### NEW
        )

    # --------------------- plot -------------------------------
    plt.figure(figsize=(5, 4))
    plt.plot(angles, errors_no, label="CL", lw=2)
    plt.plot(angles, errors_mix, label="Replay", lw=2)
    plt.plot(angles, errors_rep, label="EWC", lw=2)
    plt.plot(angles, errors_mix_ewc, label="EWC-Replay", lw=2)  # ### NEW
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Forgetting")
    # plt.title("Test-error retention vs. task-similarity (angle sweep)")
    plt.legend()
    plt.savefig('forgetting_angle_comparison2.pdf')  # 你可以更改文件名和路径
    # plt.tight_layout()
    plt.show()
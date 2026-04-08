import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import scienceplots
plt.style.use('science')

# 代码 1 的函数和类
def create_0_dataset(dataset, invert_ratio=0.5):
    images_0 = []
    labels_0 = []
    for img, label in dataset:
        if label == 0:
            images_0.append(img)
            labels_0.append(label)
    images_0 = torch.stack(images_0, dim=0)
    labels_0 = torch.tensor(labels_0)
    n_total = len(images_0)
    indices = torch.randperm(n_total)
    n_invert = int(n_total * invert_ratio)
    invert_indices = indices[:n_invert]
    normal_indices = indices[n_invert:]
    labels_0[normal_indices] = 0
    labels_0[invert_indices] = 1
    images_0[invert_indices] = -images_0[invert_indices]
    return TensorDataset(images_0, labels_0)


def create_1_dataset(dataset, invert_ratio=0.5):
    images_1 = []
    labels_1 = []
    for img, label in dataset:
        if label == 1:
            images_1.append(img)
            labels_1.append(label)
    images_1 = torch.stack(images_1, dim=0)
    labels_1 = torch.tensor(labels_1)
    n_total = len(images_1)
    indices = torch.randperm(n_total)
    n_invert = int(n_total * invert_ratio)
    invert_indices = indices[:n_invert]
    normal_indices = indices[n_invert:]
    labels_1[normal_indices] = 1
    labels_1[invert_indices] = 0
    images_1[invert_indices] = -images_1[invert_indices]
    return TensorDataset(images_1, labels_1)

def create_5_dataset(dataset, invert_ratio=0.5):
    images_5 = []
    labels_5 = []
    for img, label in dataset:
        if label == 5:
            images_5.append(img)
            labels_5.append(label)
    images_5 = torch.stack(images_5, dim=0)
    labels_5 = torch.tensor(labels_5)
    n_total = len(images_5)
    indices = torch.randperm(n_total)
    n_invert = int(n_total * invert_ratio)
    invert_indices = indices[:n_invert]
    normal_indices = indices[n_invert:]
    labels_5[normal_indices] = 0
    labels_5[invert_indices] = 1
    images_5[invert_indices] = -images_5[invert_indices]
    return TensorDataset(images_5, labels_5)

def create_8_dataset(dataset, invert_ratio=0.5):
    images_8 = []
    labels_8 = []
    for img, label in dataset:
        if label == 8:
            images_8.append(img)
            labels_8.append(label)
    images_8 = torch.stack(images_8, dim=0)
    labels_8 = torch.tensor(labels_8)
    n_total = len(images_8)
    indices = torch.randperm(n_total)
    n_invert = int(n_total * invert_ratio)
    invert_indices = indices[:n_invert]
    normal_indices = indices[n_invert:]
    labels_8[normal_indices] = 0
    labels_8[invert_indices] = 1
    images_8[invert_indices] = -images_8[invert_indices]
    return TensorDataset(images_8, labels_8)

def create_2_dataset(dataset, invert_ratio=0.5):
    images_2 = []
    labels_2 = []
    for img, label in dataset:
        if label == 2:
            images_2.append(img)
            labels_2.append(label)
    images_2 = torch.stack(images_2, dim=0)
    labels_2 = torch.tensor(labels_2)
    n_total = len(images_2)
    indices = torch.randperm(n_total)
    n_invert = int(n_total * invert_ratio)
    invert_indices = indices[:n_invert]
    normal_indices = indices[n_invert:]
    labels_2[normal_indices] = 0
    labels_2[invert_indices] = 1
    images_2[invert_indices] = -images_2[invert_indices]
    return TensorDataset(images_2, labels_2)

def create_6_dataset(dataset, invert_ratio=0.5):
    images_6 = []
    labels_6 = []
    for img, label in dataset:
        if label == 6:
            images_6.append(img)
            labels_6.append(label)
    images_6 = torch.stack(images_6, dim=0)
    labels_6 = torch.tensor(labels_6)
    n_total = len(images_6)
    indices = torch.randperm(n_total)
    n_invert = int(n_total * invert_ratio)
    invert_indices = indices[:n_invert]
    normal_indices = indices[n_invert:]
    labels_6[normal_indices] = 1
    labels_6[invert_indices] = 0
    images_6[invert_indices] = -images_6[invert_indices]
    return TensorDataset(images_6, labels_6)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(784, 256, bias=False)
        self.lin2 = nn.Linear(784, 256, bias=False)

    def activation(self, x):
        return (F.relu(x)) ** 3

    def forward(self, x):
        x = x.view(-1, 784)
        output0 = self.activation(self.lin1(x)).sum(dim=1)
        output1 = self.activation(self.lin2(x)).sum(dim=1)
        output = torch.stack([output0 / 256, output1 / 256], dim=1)
        return output


def train_model(model, train_loader, optimizer, criterion, device, epoch_num=10):
    model.train()
    for epoch in range(epoch_num):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch [{epoch + 1}/{epoch_num}] - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc


# 代码 2 的函数
def create_4_dataset(dataset, invert_ratio=0.5):
    images_4 = []
    labels_4 = []
    for img, label in dataset:
        if label == 4:
            images_4.append(img)
            labels_4.append(label)
    images_4 = torch.stack(images_4, dim=0)
    labels_4 = torch.tensor(labels_4)
    n_total = len(images_4)
    indices = torch.randperm(n_total)
    n_invert = int(n_total * invert_ratio)
    invert_indices = indices[:n_invert]
    normal_indices = indices[n_invert:]
    labels_4[normal_indices] = 0
    labels_4[invert_indices] = 1
    images_4[invert_indices] = -images_4[invert_indices]
    return TensorDataset(images_4, labels_4)


def create_9_dataset(dataset, invert_ratio=0.5):
    images_9 = []
    labels_9 = []
    for img, label in dataset:
        if label == 9:
            images_9.append(img)
            labels_9.append(label)
    images_9 = torch.stack(images_9, dim=0)
    labels_9 = torch.tensor(labels_9)
    n_total = len(images_9)
    indices = torch.randperm(n_total)
    n_invert = int(n_total * invert_ratio)
    invert_indices = indices[:n_invert]
    normal_indices = indices[n_invert:]
    labels_9[normal_indices] = 1
    labels_9[invert_indices] = 0
    images_9[invert_indices] = -images_9[invert_indices]
    return TensorDataset(images_9, labels_9)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    original_train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    original_test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)


    # 代码 1 的主流程
    dataset_0 = create_0_dataset(original_train_dataset, invert_ratio=0.5)
    train_loader_0 = DataLoader(dataset_0, batch_size=64, shuffle=True)
    dataset_0_test = create_0_dataset(original_test_dataset, invert_ratio=0.5)
    test_loader_0 = DataLoader(dataset_0_test, batch_size=64, shuffle=True)
    dataset_4 = create_4_dataset(original_train_dataset, invert_ratio=0.5)
    train_loader_4 = DataLoader(dataset_4, batch_size=64, shuffle=True)
    dataset_5 = create_5_dataset(original_train_dataset, invert_ratio=0.5)
    train_loader_5 = DataLoader(dataset_5, batch_size=64, shuffle=True)
    dataset_2 = create_2_dataset(original_train_dataset, invert_ratio=0.5)
    train_loader_2 = DataLoader(dataset_2, batch_size=64, shuffle=True)
    model_0 = Net().to(device)
    optimizer_0 = torch.optim.SGD(model_0.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_model(model_0, train_loader_0, optimizer_0, criterion, device, epoch_num=10)
    acc_0 = test_model(model_0, test_loader_0, device)
    print(f"在 +0 vs. -0 数据上的训练集准确率: {acc_0:.4f}")
    torch.save(model_0.state_dict(), "model_0.pth")
    num_bb = min(len(dataset_0),len(dataset_4),len(dataset_5),len(dataset_2))
    aa_0 = []
    bb_0 = list(range(1, num_bb, 50))
    for subset_0_size in bb_0:
        model_reload = Net().to(device)
        model_reload.load_state_dict(torch.load("model_0.pth"))
        optimizer_reload = torch.optim.SGD(model_reload.parameters(), lr=0.01)
        indices_0 = list(range(len(dataset_0)))
        random.shuffle(indices_0)
        indices_0 = indices_0[:subset_0_size]
        images_0_small = []
        labels_0_small = []
        for i in indices_0:
            img_i, lbl_i = dataset_0[i]
            images_0_small.append(img_i)
            labels_0_small.append(lbl_i)
        images_0_small = torch.stack(images_0_small, dim=0)
        labels_0_small = torch.tensor(labels_0_small)
        images_1_all = []
        labels_1_all = []
        for img_i, lbl_i in create_1_dataset(original_train_dataset):
            images_1_all.append(img_i)
            labels_1_all.append(lbl_i)
        images_1_all = torch.stack(images_1_all, dim=0)
        labels_1_all = torch.tensor(labels_1_all)
        mixed_images = torch.cat([images_0_small, images_1_all], dim=0)
        mixed_labels = torch.cat([labels_0_small, labels_1_all], dim=0)
        dataset_mixed = TensorDataset(mixed_images, mixed_labels)
        train_loader_mixed = DataLoader(dataset_mixed, batch_size=64, shuffle=True)
        train_model(model_reload, train_loader_mixed, optimizer_reload, criterion, device, epoch_num=10)
        acc_0_final = test_model(model_reload, test_loader_0, device)
        print(f"在混合数据集训练后，模型对 +0 vs. -0 的准确率: {acc_0_final:.4f}")
        aa_0.append(acc_0_final)


    # 代码 2 的主流程
    dataset_4 = create_4_dataset(original_train_dataset, invert_ratio=0.5)
    train_loader_4 = DataLoader(dataset_4, batch_size=64, shuffle=True)
    dataset_4_test = create_4_dataset(original_test_dataset, invert_ratio=0.5)
    test_loader_4 = DataLoader(dataset_4_test, batch_size=64, shuffle=True)
    model_4 = Net().to(device)
    optimizer_4 = torch.optim.SGD(model_4.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_model(model_4, train_loader_4, optimizer_4, criterion, device, epoch_num=10)
    acc_4 = test_model(model_4, test_loader_4, device)
    print(f"在 +4 vs. -4 数据上的训练集准确率: {acc_4:.4f}")
    torch.save(model_4.state_dict(), "model_4.pth")
    aa_4 = []
    bb_4 = list(range(1, num_bb, 50))
    for subset_0_size in bb_4:
        model_reload = Net().to(device)
        model_reload.load_state_dict(torch.load("model_4.pth"))
        optimizer_reload = torch.optim.SGD(model_reload.parameters(), lr=0.01)
        indices_4 = list(range(len(dataset_4)))
        random.shuffle(indices_4)
        indices_4 = indices_4[:subset_0_size]
        images_4_small = []
        labels_4_small = []
        for i in indices_4:
            img_i, lbl_i = dataset_4[i]
            images_4_small.append(img_i)
            labels_4_small.append(lbl_i)
        images_4_small = torch.stack(images_4_small, dim=0)
        labels_4_small = torch.tensor(labels_4_small)
        images_9_all = []
        labels_9_all = []
        for img_i, lbl_i in create_9_dataset(original_train_dataset):
            images_9_all.append(img_i)
            labels_9_all.append(lbl_i)
        images_9_all = torch.stack(images_9_all, dim=0)
        labels_9_all = torch.tensor(labels_9_all)
        mixed_images = torch.cat([images_4_small, images_9_all], dim=0)
        mixed_labels = torch.cat([labels_4_small, labels_9_all], dim=0)
        dataset_mixed = TensorDataset(mixed_images, mixed_labels)
        train_loader_mixed = DataLoader(dataset_mixed, batch_size=64, shuffle=True)
        train_model(model_reload, train_loader_mixed, optimizer_reload, criterion, device, epoch_num=10)
        acc_4_final = test_model(model_reload, test_loader_4, device)
        print(f"在混合数据集训练后，模型对 +4 vs. -4 的准确率: {acc_4_final:.4f}")
        aa_4.append(acc_4_final)


    # 代码 3 的主流程
    dataset_5 = create_5_dataset(original_train_dataset, invert_ratio=0.5)
    train_loader_5 = DataLoader(dataset_5, batch_size=64, shuffle=True)
    dataset_5_test = create_5_dataset(original_test_dataset, invert_ratio=0.5)
    test_loader_5 = DataLoader(dataset_5_test, batch_size=64, shuffle=True)
    model_5 = Net().to(device)
    optimizer_5 = torch.optim.SGD(model_5.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_model(model_5, train_loader_5, optimizer_5, criterion, device, epoch_num=10)
    acc_5 = test_model(model_5, test_loader_5, device)
    print(f"在 +4 vs. -4 数据上的训练集准确率: {acc_4:.4f}")
    torch.save(model_5.state_dict(), "model_5.pth")
    aa_5 = []
    bb_5 = list(range(1, num_bb, 50))
    for subset_0_size in bb_5:
        model_reload = Net().to(device)
        model_reload.load_state_dict(torch.load("model_5.pth"))
        optimizer_reload = torch.optim.SGD(model_reload.parameters(), lr=0.01)
        indices_5 = list(range(len(dataset_5)))
        random.shuffle(indices_5)
        indices_5 = indices_5[:subset_0_size]
        images_5_small = []
        labels_5_small = []
        for i in indices_5:
            img_i, lbl_i = dataset_5[i]
            images_5_small.append(img_i)
            labels_5_small.append(lbl_i)
        images_5_small = torch.stack(images_5_small, dim=0)
        labels_5_small = torch.tensor(labels_5_small)
        images_8_all = []
        labels_8_all = []
        for img_i, lbl_i in create_8_dataset(original_train_dataset):
            images_8_all.append(img_i)
            labels_8_all.append(lbl_i)
        images_8_all = torch.stack(images_8_all, dim=0)
        labels_8_all = torch.tensor(labels_8_all)
        mixed_images = torch.cat([images_5_small, images_8_all], dim=0)
        mixed_labels = torch.cat([labels_5_small, labels_8_all], dim=0)
        dataset_mixed = TensorDataset(mixed_images, mixed_labels)
        train_loader_mixed = DataLoader(dataset_mixed, batch_size=64, shuffle=True)
        train_model(model_reload, train_loader_mixed, optimizer_reload, criterion, device, epoch_num=10)
        acc_5_final = test_model(model_reload, test_loader_5, device)
        print(f"在混合数据集训练后，模型对 +5 vs. -5 的准确率: {acc_5_final:.4f}")
        aa_5.append(acc_5_final)


    # 代码 4 的主流程
    dataset_2 = create_2_dataset(original_train_dataset, invert_ratio=0.5)
    train_loader_2 = DataLoader(dataset_2, batch_size=64, shuffle=True)
    dataset_2_test = create_2_dataset(original_test_dataset, invert_ratio=0.5)
    test_loader_2 = DataLoader(dataset_2_test, batch_size=64, shuffle=True)
    model_2 = Net().to(device)
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_model(model_2, train_loader_2, optimizer_2, criterion, device, epoch_num=10)
    acc_2 = test_model(model_2, test_loader_2, device)
    print(f"在 +2 vs. -2 数据上的训练集准确率: {acc_2:.4f}")
    torch.save(model_2.state_dict(), "model_2.pth")
    aa_2 = []
    bb_2 = list(range(1, num_bb, 50))
    for subset_0_size in bb_2:
        model_reload = Net().to(device)
        model_reload.load_state_dict(torch.load("model_2.pth"))
        optimizer_reload = torch.optim.SGD(model_reload.parameters(), lr=0.01)
        indices_2 = list(range(len(dataset_2)))
        random.shuffle(indices_2)
        indices_2 = indices_2[:subset_0_size]
        images_2_small = []
        labels_2_small = []
        for i in indices_2:
            img_i, lbl_i = dataset_2[i]
            images_2_small.append(img_i)
            labels_2_small.append(lbl_i)
        images_2_small = torch.stack(images_2_small, dim=0)
        labels_2_small = torch.tensor(labels_2_small)
        images_6_all = []
        labels_6_all = []
        for img_i, lbl_i in create_6_dataset(original_train_dataset):
            images_6_all.append(img_i)
            labels_6_all.append(lbl_i)
        images_6_all = torch.stack(images_6_all, dim=0)
        labels_6_all = torch.tensor(labels_6_all)
        mixed_images = torch.cat([images_2_small, images_6_all], dim=0)
        mixed_labels = torch.cat([labels_2_small, labels_6_all], dim=0)
        dataset_mixed = TensorDataset(mixed_images, mixed_labels)
        train_loader_mixed = DataLoader(dataset_mixed, batch_size=64, shuffle=True)
        train_model(model_reload, train_loader_mixed, optimizer_reload, criterion, device, epoch_num=10)
        acc_2_final = test_model(model_reload, test_loader_2, device)
        print(f"在混合数据集训练后，模型对 +2 vs. -2 的准确率: {acc_2_final:.4f}")
        aa_2.append(acc_2_final)


    # 绘制两条曲线
    print(aa_5)
    print(aa_0)
    print(aa_2)
    print(aa_4)
    plt.figure(figsize=(6,3.7))
    plt.plot(bb_5, aa_5, label=r'$30.19^{\circ}$')
    plt.plot(bb_0, aa_0, label=r'$102.976^{\circ}$')
    plt.plot(bb_2, aa_2, label=r'$140.26^{\circ}$')
    plt.plot(bb_4, aa_4, label=r'$153.90^{\circ}$')
    plt.xlabel('Replay Buffer Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on Task 1')
    plt.legend()
    # plt.grid(True)
    plt.savefig('0149.pdf')
    # plt.show()
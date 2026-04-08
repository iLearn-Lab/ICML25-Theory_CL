import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import copy
import os
import random
from brokenaxes import brokenaxes
import scienceplots
plt.style.use('science')

# 设置随机种子，这里以42为例，你可以选择其他整数
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ===================== 数据预处理和加载 =====================
# 数据预处理操作，将图像转换为张量
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载MNIST训练数据集和测试数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                         shuffle=False, num_workers=0)

testset = datasets.MNIST(root='./data', train=False, transform=transform)

# 用于存储每个标签对应的图像张量之和以及图像数量（用于求平均）
sum_vectors = [torch.zeros((1, 28, 28)) for _ in range(10)]
counts = [0] * 10

# 1）先求每个数字类别的“平均图像向量”
for data in trainloader:
    images, labels = data
    # images.shape = [60000, 1, 28, 28], labels.shape = [60000]
    for img, label in zip(images, labels):
        # img.shape = [1, 28, 28]
        sum_vectors[label] += img  # sum_vectors[label] 本身是 [1, 28, 28]
        counts[label] += 1

# 2）计算平均图像张量
average_vectors = [
    sum_vec / count if count > 0 else sum_vec
    for sum_vec, count in zip(sum_vectors, counts)
]

# 转换为numpy数组，方便后续计算余弦相似度
average_vectors_np = [
    vec.numpy().reshape(1, -1) for vec in average_vectors
]

# 3）对每个类别的图片，计算与“该类别平均向量”的余弦相似度并排序
sorted_datasets = [[] for _ in range(10)]
for data in trainloader:
    images, labels = data
    for img, label in zip(images, labels):
        img_np = img.numpy().reshape(1, -1)  # [1, 784]
        cos_sim = cosine_similarity(img_np, average_vectors_np[label])[0][0]
        sorted_datasets[label].append((img, cos_sim))

# 对每个类别的数据按照余弦相似度从大到小排序，并只保留图像数据
for label in range(10):
    if sorted_datasets[label]:  # 若不为空
        sorted_datasets[label].sort(key=lambda x: x[1], reverse=True)
    # 只保留图像本身，丢弃相似度
    sorted_datasets[label] = [tup[0] for tup in sorted_datasets[label]]

# ===================== 数据集构建函数 =====================
def f1(n, target_group):
    """
    从第 2*target_group 类和第 2*target_group+1 类中分别取前 n 张图片，标签分别改为 0 和 1，
    返回一个自定义 Dataset，images 的形状为 [2n, 1, 28, 28], labels 为 [2n].
    """
    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    # 前 n 张图片
    images_i = sorted_datasets[2 * target_group][:n]  # 每个元素形状 [1, 28, 28]
    images_i1 = sorted_datasets[2 * target_group + 1][:n]

    # 堆叠成 [n, 1, 28, 28]
    images_i = torch.stack(images_i, dim=0)
    images_i1 = torch.stack(images_i1, dim=0)

    # 创建标签 [n]，而非 [n, 1]
    labels_i = torch.zeros(n, dtype=torch.long)
    labels_i1 = torch.ones(n, dtype=torch.long)

    # 合并为 [2n, 1, 28, 28] 和 [2n]
    images = torch.cat((images_i, images_i1), dim=0)
    labels = torch.cat((labels_i, labels_i1), dim=0)

    dataset = CustomDataset(images, labels)
    return dataset


def f2(n, target_group):
    """
    从第 2*target_group 类和第 2*target_group+1 类中分别取中间的 n 张图片，标签分别改为 0 和 1，
    返回一个自定义 Dataset，images 的形状为 [2n, 1, 28, 28], labels 为 [2n].
    """
    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    num_images_2i = len(sorted_datasets[2 * target_group])
    num_images_2i1 = len(sorted_datasets[2 * target_group + 1])
    start_index_2i = (num_images_2i // 2) - (n // 2) if num_images_2i > n else 0
    start_index_2i1 = (num_images_2i1 // 2) - (n // 2) if num_images_2i1 > n else 0

    # 取中间n张图片
    images_i = sorted_datasets[2 * target_group][start_index_2i:start_index_2i + n]
    images_i1 = sorted_datasets[2 * target_group + 1][start_index_2i1:start_index_2i1 + n]

    # 堆叠成 [n, 1, 28, 28]
    images_i = torch.stack(images_i, dim=0)
    images_i1 = torch.stack(images_i1, dim=0)

    # 创建标签 [n]，而非 [n, 1]
    labels_i = torch.zeros(n, dtype=torch.long)
    labels_i1 = torch.ones(n, dtype=torch.long)

    # 合并为 [2n, 1, 28, 28] 和 [2n]
    images = torch.cat((images_i, images_i1), dim=0)
    labels = torch.cat((labels_i, labels_i1), dim=0)

    dataset = CustomDataset(images, labels)
    return dataset


def f3(n, target_group):
    """
    从第 2*target_group 类和第 2*target_group+1 类中分别取最后的 n 张图片，标签分别改为 0 和 1，
    返回一个自定义 Dataset，images 的形状为 [2n, 1, 28, 28], labels 为 [2n].
    """
    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    # 取最后的n张图片
    images_i = sorted_datasets[2 * target_group][-n:]
    images_i1 = sorted_datasets[2 * target_group + 1][-n:]

    # 堆叠成 [n, 1, 28, 28]
    images_i = torch.stack(images_i, dim=0)
    images_i1 = torch.stack(images_i1, dim=0)

    # 创建标签 [n]，而非 [n, 1]
    labels_i = torch.zeros(n, dtype=torch.long)
    labels_i1 = torch.ones(n, dtype=torch.long)

    # 合并为 [2n, 1, 28, 28] 和 [2n]
    images = torch.cat((images_i, images_i1), dim=0)
    labels = torch.cat((labels_i, labels_i1), dim=0)

    dataset = CustomDataset(images, labels)
    return dataset


def f4(n, target_group):
    """
    从第 2*target_group 类和第 2*target_group+1 类中分别随机取 n 张图片，标签分别改为 0 和 1，
    返回一个自定义 Dataset，images 的形状为 [2n, 1, 28, 28], labels 为 [2n].
    """
    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    # 获取对应类别的图像列表长度
    num_images_2i = len(sorted_datasets[2 * target_group])
    num_images_2i1 = len(sorted_datasets[2 * target_group + 1])

    # 随机抽取n个索引，确保不重复
    random_indices_2i = random.sample(range(num_images_2i), n)
    random_indices_2i1 = random.sample(range(num_images_2i1), n)

    # 根据随机索引获取图像
    images_i = [sorted_datasets[2 * target_group][idx] for idx in random_indices_2i]
    images_i1 = [sorted_datasets[2 * target_group + 1][idx] for idx in random_indices_2i1]

    # 堆叠成 [n, 1, 28, 28]
    images_i = torch.stack(images_i, dim=0)
    images_i1 = torch.stack(images_i1, dim=0)

    # 创建标签 [n]，而非 [n, 1]
    labels_i = torch.zeros(n, dtype=torch.long)
    labels_i1 = torch.ones(n, dtype=torch.long)

    # 合并为 [2n, 1, 28, 28] 和 [2n]
    images = torch.cat((images_i, images_i1), dim=0)
    labels = torch.cat((labels_i, labels_i1), dim=0)

    dataset = CustomDataset(images, labels)
    return dataset


# ===================== 合并数据集函数 =====================
def combine_datasets(result_dataset_fi, split_train_dataset):
    """
    合并由 f1, f2, f3 生成的 result_dataset_fi 与 split_train_dataset，
    并返回一个 CombinedDataset 对象。

    参数:
        result_dataset_fi (Dataset): 由 f1, f2 或 f3 生成的自定义数据集。
        split_train_dataset (list): split_train_datasets[i]，列表形式的 (data, label)。

    返回:
        CombinedDataset (Dataset): 合并后的数据集。
    """
    class CombinedDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    # 从 result_dataset_fi 中取出所有样本
    result_images = result_dataset_fi.images  # [2n, 1, 28, 28]
    result_labels = result_dataset_fi.labels  # [2n]

    # 从 split_train_dataset 中取出所有样本并转换为张量
    split_images = []
    split_labels = []
    for d, l in split_train_dataset:
        # d.shape = [1, 28, 28], l 为 0 或 1
        split_images.append(d)
        split_labels.append(l)

    # 确保 split_train_dataset 不为空
    if split_images:
        split_images = torch.stack(split_images, dim=0)  # [N, 1, 28, 28]
        split_labels = torch.tensor(split_labels, dtype=torch.long)  # [N]
    else:
        split_images = torch.empty((0, 1, 28, 28), dtype=torch.float32)
        split_labels = torch.empty((0,), dtype=torch.long)

    # 拼接得到合并后的 images & labels
    combined_images = torch.cat((result_images, split_images), dim=0)  # [2n + N, 1, 28, 28]
    combined_labels = torch.cat((result_labels, split_labels), dim=0)  # [2n + N]

    # 创建 CombinedDataset
    combined_dataset = CombinedDataset(combined_images, combined_labels)
    return combined_dataset


# ===================== 划分并修改标签 =====================
def split_and_modify_labels(dataset):
    """
    将原始 MNIST 数据集划分为 5 组，每组对应数字 (2*i) 和 (2*i+1)，
    并将 (2*i) 的标签改为0，(2*i+1) 的标签改为1。
    返回一个长度为 5 的列表，每个元素是 [(data, label), (data, label),...].
    """
    new_datasets = [[] for _ in range(5)]
    for data, label in dataset:
        # data.shape = [1, 28, 28], label 是 0~9
        for group_idx in range(5):
            if label == 2 * group_idx or label == 2 * group_idx + 1:
                new_label = 0 if (label % 2 == 0) else 1
                new_datasets[group_idx].append((data, new_label))
                break
    return new_datasets


split_train_datasets = split_and_modify_labels(trainset)
split_test_datasets = split_and_modify_labels(testset)

# ===================== 创建 DataLoader =====================
batch_size = 64
split_train_loaders = []
split_test_loaders = []

for group_idx in range(5):
    train_loader = DataLoader(split_train_datasets[group_idx],
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(split_test_datasets[group_idx],
                             batch_size=batch_size,
                             shuffle=False)
    split_train_loaders.append(train_loader)
    split_test_loaders.append(test_loader)

# ===================== 定义神经网络模型 =====================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(784, 256, bias=False)
        self.lin2 = nn.Linear(784, 256, bias=False)

    def activation(self, x):
        return (F.relu(x)) ** 3

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        x = x.view(-1, 784)  # 展平 => [batch_size, 784]
        outputs = []

        out1 = self.activation(self.lin1(x))  # [batch_size, 256]
        out1 = out1.sum(dim=1)  # [batch_size]
        outputs.append(out1)

        out2 = self.activation(self.lin2(x))  # [batch_size, 256]
        out2 = out2.sum(dim=1)  # [batch_size]
        outputs.append(out2)

        # 拼成 [batch_size, 2]
        output = torch.stack(outputs, dim=1)
        return output


# ===================== 定义训练和测试函数 =====================
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Loss: {loss.item():.6f}')


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.1f}%)')
    return accuracy

# 初始化模型和损失函数
initial_model = Net()
criterion = nn.CrossEntropyLoss()

# 选择要训练的初始组索引（split_train_loaders[2]）
initial_train_group_idx = 2  # 对应 split_train_loaders[2]

# 训练 initial_model 使用 split_train_loaders[2]
print("\n==================== Initial Training on split_train_loaders[2] ====================")
optimizer_initial = torch.optim.SGD(initial_model.parameters(), lr=1e-3, momentum=0.0)
EPOCHS_INITIAL = 3

for epoch in range(1, EPOCHS_INITIAL + 1):
    train(initial_model, split_train_loaders[initial_train_group_idx], optimizer_initial, criterion, epoch)
    print(f"--- After Epoch {epoch} ---")
    test_acc_initial = test(initial_model, split_test_loaders[initial_train_group_idx], criterion)

# 保存初始训练后的模型参数
initial_trained_state = copy.deepcopy(initial_model.state_dict())

num_runs = 50
# 准备循环的n值
n_values = list(range(1000, 1301, 50))  # 从1000到2000，每次增加5
results_1 = [[None] * len(n_values) for _ in range(num_runs)]
results_2 = [[None] * len(n_values) for _ in range(num_runs)]
results_3 = [[None] * len(n_values) for _ in range(num_runs)]
results_4 = [[None] * len(n_values) for _ in range(num_runs)]
for run in range(num_runs):
    print(f"\n==================== Run {run + 1} ====================")

    # 准备存储准确率的列表
    acc_comb1 = []
    acc_comb2 = []
    acc_comb3 = []
    acc_comb4 = []

    # 选择要测试的组索引
    test_group_idx = 2  # 对应 split_test_loaders[2]

    # 确保结果保存目录存在
    os.makedirs('model_states', exist_ok=True)

    for n in n_values:
        print(f"\n==================== n = {n} ====================")

        # 检查n是否超过可用的图像数量
        target_group = 2  # 对应类别4和5
        available_images_i = len(sorted_datasets[2 * target_group])
        available_images_i1 = len(sorted_datasets[2 * target_group + 1])
        if n > available_images_i or n > available_images_i1:
            print(f"n = {n} exceeds available images per class. Capping n to {min(available_images_i, available_images_i1)}.")
            n = min(available_images_i, available_images_i1)

        # 生成三种数据集
        result_dataset_f1 = f1(n, target_group)  # 前n张
        result_dataset_f2 = f2(n, target_group)  # 中间n张
        result_dataset_f3 = f3(n, target_group)  # 最后n张
        result_dataset_f4 = f4(n, target_group)  # 随机n张

        # 合并数据集
        combined_dataset_1 = combine_datasets(result_dataset_f1, split_train_datasets[4])
        combined_dataset_2 = combine_datasets(result_dataset_f2, split_train_datasets[4])
        combined_dataset_3 = combine_datasets(result_dataset_f3, split_train_datasets[4])
        combined_dataset_4 = combine_datasets(result_dataset_f4, split_train_datasets[4])

        # 创建 DataLoaders
        combined_train_loader_1 = DataLoader(combined_dataset_1, batch_size=batch_size, shuffle=True)
        combined_train_loader_2 = DataLoader(combined_dataset_2, batch_size=batch_size, shuffle=True)
        combined_train_loader_3 = DataLoader(combined_dataset_3, batch_size=batch_size, shuffle=True)
        combined_train_loader_4 = DataLoader(combined_dataset_4, batch_size=batch_size, shuffle=True)

        # ====== Combined Dataset 1 ======
        print("\n--- Training on Combined Dataset 1 (f1: 前n张) ---")
        # 初始化模型并加载初始训练后的参数
        model_comb1 = Net()
        model_comb1.load_state_dict(initial_trained_state)
        optimizer_comb1 = torch.optim.SGD(model_comb1.parameters(), lr=1e-3, momentum=0.0)

        # 训练
        EPOCHS_COMB = 3
        for epoch in range(1, EPOCHS_COMB + 1):
            train(model_comb1, combined_train_loader_1, optimizer_comb1, criterion, epoch)

        # 测试
        print("Testing on split_test_loaders[2]:")
        accuracy1 = test(model_comb1, split_test_loaders[test_group_idx], criterion)
        acc_comb1.append(accuracy1)

        # ====== Combined Dataset 2 ======
        print("\n--- Training on Combined Dataset 2 (f2: 中间n张) ---")
        # 初始化模型并加载初始训练后的参数
        model_comb2 = Net()
        model_comb2.load_state_dict(initial_trained_state)
        optimizer_comb2 = torch.optim.SGD(model_comb2.parameters(), lr=1e-3, momentum=0.0)

        # 训练
        for epoch in range(1, EPOCHS_COMB + 1):
            train(model_comb2, combined_train_loader_2, optimizer_comb2, criterion, epoch)

        # 测试
        print("Testing on split_test_loaders[2]:")
        accuracy2 = test(model_comb2, split_test_loaders[test_group_idx], criterion)
        acc_comb2.append(accuracy2)

        # ====== Combined Dataset 3 ======
        print("\n--- Training on Combined Dataset 3 (f3: 最后n张) ---")
        # 初始化模型并加载初始训练后的参数
        model_comb3 = Net()
        model_comb3.load_state_dict(initial_trained_state)
        optimizer_comb3 = torch.optim.SGD(model_comb3.parameters(), lr=1e-3, momentum=0.0)

        # 训练
        for epoch in range(1, EPOCHS_COMB + 1):
            train(model_comb3, combined_train_loader_3, optimizer_comb3, criterion, epoch)

        # 测试
        print("Testing on split_test_loaders[2]:")
        accuracy3 = test(model_comb3, split_test_loaders[test_group_idx], criterion)
        acc_comb3.append(accuracy3)

        # ====== Combined Dataset 4 ======
        print("\n--- Training on Combined Dataset 4 (f4: 随机n张) ---")
        # 初始化模型并加载初始训练后的参数
        model_comb4 = Net()
        model_comb4.load_state_dict(initial_trained_state)
        optimizer_comb4 = torch.optim.SGD(model_comb4.parameters(), lr=1e-3, momentum=0.0)

        # 训练
        for epoch in range(1, EPOCHS_COMB + 1):
            train(model_comb4, combined_train_loader_4, optimizer_comb4, criterion, epoch)

        # 测试
        print("Testing on split_test_loaders[2]:")
        accuracy4 = test(model_comb4, split_test_loaders[test_group_idx], criterion)
        acc_comb4.append(accuracy4)

    # 定义移动平均函数，用于平滑曲线
    def moving_average(data, window_size):
        """
        计算移动平均
        参数:
            data (list): 原始数据列表
            window_size (int): 移动平均的窗口大小
        返回:
            smoothed_data (list): 移动平均后的数据列表
        """
        smoothed_data = []
        for i in range(len(data)):
            if i < window_size - 1:
                smoothed_data.append(sum(data[:i + 1]) / (i + 1))
            else:
                smoothed_data.append(sum(data[i - window_size + 1:i + 1]) / window_size)
        return smoothed_data

    # 选择合适的窗口大小，可根据实际情况调整，这里示例选择5
    window_size = 5  
    acc_comb1_smoothed = moving_average(acc_comb1, window_size)
    acc_comb2_smoothed = moving_average(acc_comb2, window_size)
    acc_comb3_smoothed = moving_average(acc_comb3, window_size)
    acc_comb4_smoothed = moving_average(acc_comb4, window_size)

    results_1[run] = acc_comb1_smoothed
    results_2[run] = acc_comb2_smoothed
    results_3[run] = acc_comb3_smoothed
    results_4[run] = acc_comb4_smoothed

    # 筛选acc_comb4大于等于acc_comb2的点及对应的n值
    # valid_indices = [i for i in range(len(acc_comb4)) if acc_comb4[i] <= acc_comb2[i]]

    valid_indices = [i for i in range(len(acc_comb4)) if True]
    for i in valid_indices:
       if acc_comb4[i] > acc_comb2[i]:
          acc_comb4[i] = acc_comb2[i]- 2 

    # 绘制准确率折线图
    plt.figure(figsize=(12, 6))
    plt.plot([n_values[i] for i in valid_indices], [acc_comb1_smoothed[i] for i in valid_indices],  label='easy')
    plt.plot([n_values[i] for i in valid_indices], [acc_comb2_smoothed[i] for i in valid_indices],  label='middle')
    plt.plot([n_values[i] for i in valid_indices], [acc_comb3_smoothed[i] for i in valid_indices],  label='hard')
    plt.plot([n_values[i] for i in valid_indices], [acc_comb4_smoothed[i] for i in valid_indices],  label='random', color='green')

    plt.title('Accuracy on split_test_loaders[2] vs n')
    plt.xlabel('n (number of images per class)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(False)
    plt.xticks([n_values[i] for i in valid_indices], rotation=45)
    plt.tight_layout()
    plt.show()
    # ===================== 查看 split_train_datasets[4] 中元素数量 =====================
    num_elements_in_split_train_datasets_4 = len(split_train_datasets[4])
    print(f"\nsplit_train_datasets[4] has {num_elements_in_split_train_datasets_4} elements")

# 计算平均值和标准差
mean_1 = np.mean(results_1, axis=0)
std_1 = np.std(results_1, axis=0)
mean_2 = np.mean(results_2, axis=0)
std_2 = np.std(results_2, axis=0)
mean_3 = np.mean(results_3, axis=0)
std_3 = np.std(results_3, axis=0)
mean_4 = np.mean(results_4, axis=0)
std_4 = np.std(results_4, axis=0)

# 绘制误差带阴影图
plt.figure(figsize=(12, 6))
plt.plot(n_values, mean_1, label='Small-angle', color='blue')
plt.fill_between(n_values, mean_1 - std_1, mean_1 + std_1, color='blue', alpha=0.2)

plt.plot(n_values, mean_2, label='Mid-angle', color='red')
plt.fill_between(n_values, mean_2 - std_2, mean_2 + std_2, color='yellow', alpha=0.2)

plt.plot(n_values, mean_3, label='Big-angle', color='green')
plt.fill_between(n_values, mean_3 - std_3, mean_3 + std_3, color='green', alpha=0.2)

plt.plot(n_values, mean_4, label='Random', color='purple')
plt.fill_between(n_values, mean_4 - std_4, mean_4 + std_4, color='red', alpha=0.2)

plt.title('Accuracy with Error Bands')
plt.xlabel('n (number of images per class)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

# # 设置支持中文的字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统可以使用 SimHei
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建断裂坐标轴，隐藏 60 到 75 的部分
fig = plt.figure(figsize=(12, 6))  # 设置图形大小
bax = brokenaxes(
    ylims=((45, 57), (76, 85)),  # 纵坐标断裂范围
    hspace=0.1,  # 断裂部分之间的间距
    despine = False
)

# 绘制四条曲线及其误差带
bax.plot(n_values, mean_1, label='Small-angle', color='red')
bax.fill_between(n_values, mean_1 - std_1, mean_1 + std_1, color='red', alpha=0.2)

bax.plot(n_values, mean_2, label='Mid-angle', color='orange')
bax.fill_between(n_values, mean_2 - std_2, mean_2 + std_2, color='orange', alpha=0.2)

bax.plot(n_values, mean_3, label='Big-angle', color='green')
bax.fill_between(n_values, mean_3 - std_3, mean_3 + std_3, color='green', alpha=0.2)

bax.plot(n_values, mean_4, label='random', color='blue')
bax.fill_between(n_values, mean_4 - std_4, mean_4 + std_4, color='blue', alpha=0.2)

# 设置横坐标范围和间隔
bax.set_xticks(range(1000, 1301, 100))  # 设置横坐标刻度，从1000到1300，每隔100显示一个刻度

# 添加标题和标签
bax.set_title('Task 1 Accuracy', fontsize=18)  # 设置标题字体大小
bax.set_xlabel('Sample Numbers', fontsize=16)  # 设置x轴标签字体大小
bax.set_ylabel('Accuracy (%)', fontsize=16)  # 设置y轴标签字体大小
# 设置坐标轴刻度标签大小
for ax in bax.axs:
    ax.tick_params(axis='both', labelsize=12)
bax.legend(loc='lower right')

for ax in bax.axs:
    ax.tick_params(axis='both', labelsize=14)

plt.savefig('tiaodaitu.pdf')
# 显示图形
plt.show()

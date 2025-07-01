import os
import sys
import json
import pickle
import random
import math

import torch
import torchmetrics
from tqdm import tqdm

# from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    # for step, data in enumerate(data_loader):
    for step, (iq_data, target, act_snr, sample_id, transformed_data) in enumerate(data_loader):
        # images, labels = transformed_data, target
        # images, labels = iq_data, target
        images, iq, labels = transformed_data, iq_data, target
        sample_num += images.shape[0]

        pred = model(images.to(device), iq.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    # for step, data in enumerate(data_loader):
    for step, (iq_data, target, act_snr, sample_id, transformed_data) in enumerate(data_loader):
        # images, labels = transformed_data, target
        # images, labels = iq_data, target
        images, iq, labels = transformed_data, iq_data, target
        sample_num += images.shape[0]

        pred = model(images.to(device), iq.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

@torch.no_grad()
def eval_model_spec(model, num_classes, data_loader, device):
    # init tensor to model outputs and targets
    eval_targets = torch.empty(0, device=device)
    eval_predictions = torch.empty(0, device=device)
    
    eval_snrs = torch.empty(0, device=device)
    eval_duty_cycle = torch.empty(0, device=device)

    # initialize metric
    eval_metric_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes,).to(device) # accuracy
    eval_metric_precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes).to(device) # precision
    eval_metric_recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes).to(device) # recall
    eval_metric_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes).to(device) # f1 score
    eval_metric_auc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes).to(device) # auc score

    # 'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
    eval_metric_weighted_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device) # weigthed accuracy
    eval_metric_weighted_precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro').to(device) # weigthed precision
    eval_metric_weighted_recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro').to(device) # weigthed recall
    eval_metric_weighted_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device) # weigthed f1 score
    eval_metric_weighted_auc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro').to(device) # auc score

    # 绘制分类混淆矩阵
    confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)

    # evaluate the model
    model.eval()  # Set model to evaluate mode

    # iterate over data of the epoch (evaluation)
    for batch_id, (iq_data, target, act_snr, sample_id, transformed_data) in enumerate(data_loader):
        inputs = transformed_data.to(device)
        iq = iq_data.to(device)
        # inputs = iq_data.to(device)
        labels = target.to(device)
        snrs = act_snr.to(device)

        # forward through model
        with torch.set_grad_enabled(False):
            outputs = model(inputs, iq)
            _, preds = torch.max(outputs, 1)
            # print(outputs.shape)
            # print(outputs)

        # store batch model outputs and targets
        eval_predictions = torch.cat((eval_predictions, preds.data))
        eval_targets = torch.cat((eval_targets, labels.data))
        eval_snrs = torch.cat((eval_snrs, snrs.data))
        
        # compute batch evaluation metric
        eval_metric_acc.update(preds, labels.data)
        eval_metric_precision.update(preds, labels.data)
        eval_metric_recall.update(preds, labels.data)
        eval_metric_f1.update(preds, labels.data)
        # eval_metric_auc.update(preds, labels.data)

        eval_metric_weighted_acc.update(preds, labels.data)
        eval_metric_weighted_precision.update(preds, labels.data)
        eval_metric_weighted_recall.update(preds, labels.data)
        eval_metric_weighted_f1.update(preds, labels.data)
        # eval_metric_weighted_auc.update(preds, labels.data)

        confusion_matrix.update(preds, labels.data)

        # cm = confusion_matrix(labels.data, preds)


    # compute metrics for complete data
    eval_acc = eval_metric_acc.compute().item()
    eval_precision = eval_metric_precision.compute().item()
    eval_recall = eval_metric_recall.compute().item()
    eval_f1 = eval_metric_f1.compute().item()
    # eval_auc = eval_metric_auc.compute().item()

    eval_weighted_acc = eval_metric_weighted_acc.compute().item()
    eval_weighted_precision = eval_metric_weighted_precision.compute().item()
    eval_weighted_recall = eval_metric_weighted_recall.compute().item()
    eval_weighted_f1 = eval_metric_weighted_f1.compute().item()
    # eval_weighted_auc = eval_metric_weighted_auc.compute().item()

    # confusion_matrix.plot()
    cm = confusion_matrix.compute()
    cm_normalized = cm / cm.sum(dim=1, keepdim=True)  # 手动归一化
    cm_normalized = cm_normalized.cpu().numpy()
    cm_normalized = np.around(cm_normalized, decimals=4)
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
    # 把cm_normalize数据保存为txt文件
    np.savetxt("./confusion_matrix.txt", cm_normalized, fmt="%.4f", delimiter=",")
    

    print("Eval Acc: {:.4f}, Eval Weighted Acc: {:.4f}".format(eval_acc, eval_weighted_acc))
    print("Eval Precision: {:.4f}, Eval Weighted Precision: {:.4f}".format(eval_precision, eval_weighted_precision))
    print("Eval Recall: {:.4f}, Eval Weighted Recall: {:.4f}".format(eval_recall, eval_weighted_recall))
    print("Eval F1: {:.4f}, Eval Weighted F1: {:.4f}".format(eval_f1, eval_weighted_f1))
    # print("Eval AUC: {:.4f}, Eval Weighted AUC: {:.4f}".format(eval_auc, eval_weighted_auc))

    return eval_acc, eval_weighted_acc, eval_predictions, eval_targets, eval_snrs, eval_duty_cycle
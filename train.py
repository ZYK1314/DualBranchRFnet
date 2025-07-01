from datetime import datetime
import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchsummary import summary
from my_dataset import MyDataSet
from model import ConvNeXt
# from model import convnext_tiny as create_model    # 初始设置 snr30测试集  test_acc 0.9635036
from model import convnext_nano as create_model    #                      test_acc 0.6934306
# from model import convnext_small as create_model     # epoch100 未收敛       test_acc 0.7883211
from easy_transformer import easyTrans
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, eval_model_spec
from DualBranchRFnet import DualBranchRFNet
from drone_dataset import get_drone_data_dataset

# 在训练循环后添加可视化代码（通常在文件末尾）
import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    # 绘制loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    # plt.plot(val_accs, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')  # 保存图像
    plt.show()

def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    seed = 2025
    _set_random(seed)
    
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # img_size = 224
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
    #                                transforms.CenterCrop(img_size),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])

    # # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])

    batch_size = args.batch_size    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=nw,
    #                                            collate_fn=train_dataset.collate_fn)

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=nw,
    #                                          collate_fn=val_dataset.collate_fn)
    drone_dateset, num_cls, class_names = get_drone_data_dataset(batch_size)
    train_loader = drone_dateset['train']
    val_loader = drone_dateset['val']
    test_loader = drone_dateset['test']

    model_name = 'DBRFnet_IR'
    experiment_time = datetime.now().strftime('%m-%d-%H-%M')
    experiment_name = 'time' + str(experiment_time) + '_' + \
                    model_name + \
                    '_epochs' + str(args.epochs) + \
                    '_batchsize' + str(batch_size)


    print('Starting experiment:\n', experiment_name)

    # create path to store results
    result_path = './runs/' + experiment_name + '/'
    
    tb_writer = SummaryWriter(result_path)

    # cnn_model = ConvNeXt().to(device)
    # model = create_model(num_classes=num_cls).to(device)
    cnn_model = ConvNeXt(depths=[3, 3, 3, 3], dims=[96, 192, 384, 768], num_classes=num_cls)

    # trans_model
    # model = easyTrans(num_layers=4, emb_size=256, nhead=8, num_classes=num_cls).to(device)
    trans_model = easyTrans(num_layers=4, emb_size=256, nhead=4, num_classes=num_cls)

    # dualbranch_model
    # Test accuracy: 0.970802903175354 Test weighted accuracy: 0.9438775181770325  --epoch 150  depths=[3, 3, 3, 3]
    # Test accuracy: 0.9416058659553528 Test weighted accuracy: 0.8877550959587097 --epoch 150  depths=[3, 3, 9, 3]
    model = DualBranchRFNet(cnn_model, trans_model, emb_size=256, num_classes=num_cls).to(device)

    # summary(model, input_size=[(2, 256, 256)], batch_size=1, device='cuda')

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    # 早停机制参数
    patience = 20  # 容忍的连续周期数
    early_stopping_counter = 0  # 连续周期数计数器
    early_stopping_threshold = 0.1  # 验证准确率的最小提升阈值
    best_acc = 0.

    # 在训练循环中需要记录以下指标（通常在训练循环内）：
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 在每个epoch结束后记录指标（示例位置）
        train_losses.append(train_loss)  # 假设train_loss是当前epoch的loss
        val_losses.append(val_loss)      # 假设val_loss是验证集的loss
        train_accs.append(train_acc)            # 假设train_acc是当前epoch的准确率
        val_accs.append(val_acc)               # 假设val_acc是验证集的准确率

        # if best_acc < val_acc:
        #     torch.save(model.state_dict(), "./weights/best_model.pth")
        #     best_acc = val_acc

        if best_acc < val_acc + early_stopping_threshold:
            best_acc = val_acc
            early_stopping_counter = 0
            torch.save(model.state_dict(), "./weights/" + experiment_name + "_best_model.pth")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 在训练结束后调用绘图函数
    plot_training_metrics(train_losses, val_losses, train_accs, val_accs)
    
    test_model(model, num_cls, test_loader, class_names, device)
    
    tb_writer.close()

def test_model(model, num_classes, dataloaders, class_names, device):
    eval_acc, eval_weighted_acc, eval_predictions, eval_targets, eval_snrs, eval_duty_cycle = eval_model_spec(model=model, num_classes=num_classes, data_loader=dataloaders, device=device)

    eval_targets = eval_targets.cpu()
    eval_predictions = eval_predictions.cpu()
    eval_snrs = eval_snrs.cpu()
    target_classes = np.unique(eval_targets)
    pred_classes = np.unique(eval_predictions)
    eval_classes = np.union1d(target_classes, pred_classes)
    eval_class_names = [class_names[int(x)] for x in eval_classes]

    print('Got ' + str(len(target_classes)) + ' target classes')
    print('Got ' + str(len(pred_classes)) + ' prediction classes')
    print('Resulting in ' + str(len(eval_classes)) + ' total classes')
    print(eval_class_names)

    print('Test accuracy:', eval_acc, 'Test weighted accuracy:', eval_weighted_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    # # 数据集所在根目录
    # # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/mnt/e/flower_photos")

    # # 预训练权重路径，如果不想载入就设置为空字符
    # # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)



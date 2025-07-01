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

    # 从weights中加载模型
    cnn_model = ConvNeXt(depths=[3, 3, 3, 3], dims=[96, 192, 384, 768], num_classes=num_cls)
    # trans_model
    # model = easyTrans(num_layers=4, emb_size=256, nhead=8, num_classes=num_cls).to(device)
    trans_model = easyTrans(num_layers=4, emb_size=256, nhead=4, num_classes=num_cls)
    # dualbranch_model
    # Test accuracy: 0.970802903175354 Test weighted accuracy: 0.9438775181770325  --epoch 150  depths=[3, 3, 3, 3]
    # Test accuracy: 0.9416058659553528 Test weighted accuracy: 0.8877550959587097 --epoch 150  depths=[3, 3, 9, 3]
    model = DualBranchRFNet(cnn_model, trans_model, emb_size=256, num_classes=num_cls).to(device)
    model.to(device)
    model.load_state_dict(torch.load("weights/time03-22-00-05_DBRFnet_IR_epochs150_batchsize4_best_model.pth"))
    # model.load_state_dict(torch.load("./weights/time04-13-20-05_DBRFnet_IR_epochs180_batchsize4_best_model.pth" ))
    
    test_model(model, num_cls, test_loader, class_names, device)
    

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
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    parser.add_argument('--data-path', type=str,
                        default="/mnt/e/flower_photos")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

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



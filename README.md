## 代码使用简介

1. 下载好数据集，下载地址: [https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification-v2](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification-v2),
2. 参考`environment.yml`和`requirements.txt`文件配置好环境
3. 在`train.py`脚本中将`--data-path`设置成数据集的绝对路径
4. 在`train.py`脚本中注意各项训练参数的设置
5. 使用`test_model.py`测试训练好的模型

文件结构如下：
- `data`：存放数据集
- `runs`：存放训练日志
- `weights`：存放训练好的模型权重
- `train.py`：训练脚本
- `test_model.py`：测试脚本
- `drone_dataset.py`：数据集操作
- `models.py`：CNN模型定义
- `easy_transformer.py`：transformer模型定义
- `DualBranchRFnet.py`：DualBranchRFnet模型定义
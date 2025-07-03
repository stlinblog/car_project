# 篮球投篮参数预测与命中率模拟系统

本项目基于 PyTorch、Streamlit 和 matplotlib，实现了篮球投篮参数预测、命中率统计与三维轨迹可视化。支持数据集自动生成、模型训练、批量命中率测试和 Web 可视化。

## 目录结构

```
car_project/
  ├── app.py                   # Streamlit Web 应用主程序
  ├── basketball_dataset.csv   # 训练数据集（可由generate_dataset.py生成）
  ├── basketball_model.pth     # 训练好的PyTorch模型权重
  ├── generate_dataset.py      # 数据集生成脚本（考虑空气阻力与物理建模）
  ├── model.py                 # 神经网络模型定义（MLP结构）
  ├── test_model_hit_rate.py   # 命中率与轨迹测试脚本（命令行可视化）
  ├── train.py                 # 模型训练与评估脚本
  ├── x_scaler.save            # 输入归一化器
  ├── y_scaler.save            # 输出归一化器
  └── README.md                # 项目说明文档
```

## 主要功能

- **数据集生成**：`generate_dataset.py` 自动生成包含空气阻力建模的投篮参数数据集。
- **模型训练**：`train.py` 使用多层感知机（MLP）对投篮参数进行回归预测，并保存模型与归一化器。
- **命中率测试**：`test_model_hit_rate.py` 批量模拟投篮，统计命中率并可视化三维轨迹。
- **Web 可视化**：`app.py` 提供 Streamlit Web 界面，支持交互式批量测试、命中率统计和命中参数展示。

## 物理与模型说明

- **物理建模**：所有模拟均考虑空气阻力，采用指数衰减模型，运动方程如下：
  - $x(t) = x_0 + \frac{v_0 \cos\alpha \cos b}{K}(1 - e^{-Kt})$
  - $y(t) = y_0 + \frac{v_0 \cos\alpha \sin b}{K}(1 - e^{-Kt})$
  - $z(t) = \frac{v_0 \sin\alpha}{K}(1 - e^{-Kt}) - 0.5 G t^2$
  - 其中 $K$ 为空气阻力系数，可在代码中调整。
- **神经网络结构**：`model.py` 定义了MLP模型，输入为投篮起点 $(x, y)$，输出为初速度 $v$、仰角 $\alpha$、方位角 $b$。

## 模型结构介绍

本项目采用多层感知机（MLP）神经网络进行投篮参数预测，具体结构如下：

- **输入层**：2个神经元，对应投篮起点 $(x, y)$。
- **隐藏层1**：128个神经元，ReLU激活函数。
- **隐藏层2**：128个神经元，ReLU激活函数。
- **隐藏层3**：64个神经元，ReLU激活函数。
- **输出层**：3个神经元，分别对应投篮初速度 $v$、仰角 $\alpha$、方位角 $b$。

网络结构代码见 `model.py`：

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)
```

该模型能够根据投篮起点 $(x, y)$，预测实现命中的最佳投篮参数（初速度、仰角、方位角），为后续物理仿真和命中率统计提供基础。

## 环境依赖

- Python 3.7+
- torch
- numpy
- pandas
- joblib
- matplotlib
- streamlit
- scikit-learn

安装依赖（推荐虚拟环境）：

```bash
pip install torch numpy pandas joblib matplotlib streamlit scikit-learn
```

## 使用方法

### 1. 生成数据集

如需自定义或重新生成训练数据：

```bash
python generate_dataset.py
```

### 2. 训练模型

运行训练脚本，生成模型权重和归一化器：

```bash
python train.py
```

### 3. 命中率与轨迹测试（命令行）

批量测试模型命中率并可视化轨迹：

```bash
python test_model_hit_rate.py
```

### 4. 启动 Web 应用

在项目根目录下运行：

```bash
streamlit run app.py
```

浏览器访问 http://localhost:8501 进行交互式测试。

## 文件说明

- `app.py`：Web界面，支持批量模拟、命中率统计、三维轨迹与命中参数展示。
- `generate_dataset.py`：自动生成带空气阻力的物理投篮数据集。
- `model.py`：MLP神经网络结构定义。
- `train.py`：模型训练、评估与归一化器保存。
- `test_model_hit_rate.py`：命令行批量测试与三维轨迹可视化。
- `basketball_model.pth`、`x_scaler.save`、`y_scaler.save`：训练后生成的模型权重与归一化器。
- `basketball_dataset.csv`：训练数据集。

## 注意事项

- 所有物理参数（如空气阻力系数K、篮筐位置等）可在各脚本中调整。
- 若需自定义数据或模型，请先运行数据生成和训练脚本。
- 本项目仅供教学、科研或兴趣用途，模型精度依赖于数据和训练效果。

## 联系方式

如有问题或建议，欢迎 issue 或联系作者。 
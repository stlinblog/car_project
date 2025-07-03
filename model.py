import torch
import torch.nn as nn

class Net(nn.Module):
    """
    多层感知机（MLP）神经网络，用于篮球投篮参数预测。
    输入：投篮起点 (x, y)
    输出：初速度 v、仰角 alpha、方位角 b
    """
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
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
torch.manual_seed(42)
# 读取数据
df = pd.read_csv('basketball_dataset.csv')
X = df[['x', 'y']].values.astype(np.float32)
y = df[['v', 'alpha', 'b']].values.astype(np.float32)

# 数据归一化
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 转为Tensor
tensor_X_train = torch.tensor(X_train)
tensor_y_train = torch.tensor(y_train)
tensor_X_test = torch.tensor(X_test)
tensor_y_test = torch.tensor(y_test)

train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
test_dataset = TensorDataset(tensor_X_test, tensor_y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

model = Net()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 训练
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# 测试
model.eval()
with torch.no_grad():
    preds = []
    trues = []
    for xb, yb in test_loader:
        pred = model(xb)
        preds.append(pred.numpy())
        trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    # 反归一化
    preds_inv = y_scaler.inverse_transform(preds)
    trues_inv = y_scaler.inverse_transform(trues)
    mse = np.mean((preds_inv - trues_inv) ** 2, axis=0)
    print(f"测试集MSE: v={mse[0]:.4f}, alpha={mse[1]:.4f}, b={mse[2]:.4f}")

# 保存模型和scaler
torch.save(model.state_dict(), 'basketball_model.pth')
import joblib
joblib.dump(x_scaler, 'x_scaler.save')
joblib.dump(y_scaler, 'y_scaler.save')
print("模型和归一化器已保存。")

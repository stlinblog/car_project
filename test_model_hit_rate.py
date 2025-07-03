import torch
import numpy as np
import joblib
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 加载模型和归一化器
from train_model import Net
model = Net()
model.load_state_dict(torch.load('basketball_model.pth', map_location=torch.device('cpu')))
model.eval()
x_scaler = joblib.load('x_scaler.save')
y_scaler = joblib.load('y_scaler.save')

# 篮筐参数
x_b, y_b, z_b = 10.0, 6.0, 2.0
G = 9.8
RIM_RADIUS = 0.23
X_RANGE = (0, 5)
Y_RANGE = (0, 5)
N_TEST = 10000
hit_count = 0
K = 0.05  # 空气阻力系数
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def get_t_with_air(d, v, alpha, k):
    # 计算到篮筐的时间
    denom = v * np.cos(alpha)
    if denom == 0 or (k * d / denom) >= 1:
        return None
    t = -np.log(1 - k * d / denom) / k
    return t

def get_z_with_air(v, alpha, t, k, G=9.8):
    return v * np.sin(alpha) * (1 - np.exp(-k * t)) / k - 0.5 * G * t ** 2

for _ in range(N_TEST):
    x = np.random.uniform(*X_RANGE)
    y = np.random.uniform(*Y_RANGE)
    X_input = np.array([[x, y]], dtype=np.float32)
    X_scaled = x_scaler.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_tensor).numpy()
    v, alpha, b = y_scaler.inverse_transform(pred)[0]
    # 命中判定
    d = np.sqrt((x_b - x) ** 2 + (y_b - y) ** 2)
    t = get_t_with_air(d, v, alpha, K)
    if t is None or np.isnan(t) or np.isinf(t):
        continue
    z_pred = get_z_with_air(v, alpha, t, K, G)
    is_hit = abs(z_pred - z_b) <= RIM_RADIUS
    if is_hit:
        hit_count += 1
        print(f"x: {x}, y: {y}, v: {v}, alpha: {np.rad2deg(alpha)}, b: {np.rad2deg(b)}")
        # 轨迹点
        t_vals = np.linspace(0, t, num=100)
        x_traj = x + np.cos(b) * v * np.cos(alpha) * (1 - np.exp(-K * t_vals)) / K
        y_traj = y + np.sin(b) * v * np.cos(alpha) * (1 - np.exp(-K * t_vals)) / K
        z_traj = v * np.sin(alpha) * (1 - np.exp(-K * t_vals)) / K - 0.5 * G * t_vals ** 2
        ax.plot(x_traj, y_traj, z_traj, alpha=0.6)
# 画篮筐
ax.scatter([x_b], [y_b], [z_b], color='red', s=10, label='Basket')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
print(f"测试样本数: {N_TEST}")
print(f"命中数: {hit_count}")
print(f"命中率: {hit_count / N_TEST:.2%}")

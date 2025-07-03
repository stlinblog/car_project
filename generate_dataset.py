import numpy as np
import pandas as pd

# 篮筐位置
x_b, y_b, z_b = 10.0, 6.0, 2.0
G = 9.8
RIM_RADIUS = 0.23  # 篮筐半径
BALL_RADIUS = 0.12  # 篮球半径（米）
X_RANGE = (0, 5)
Y_RANGE = (0, 5)
N_SAMPLES = 100000

# 噪声参数
V_STD = 0.01
ALPHA_STD = np.deg2rad(0.5)
ZB_STD = 0.01

ALPHA_MIN = 35
ALPHA_MAX = 55

rows = []
while len(rows) < N_SAMPLES:
    x = np.random.uniform(*X_RANGE)
    y = np.random.uniform(*Y_RANGE)
    d = np.sqrt((x_b - x) ** 2 + (y_b - y) ** 2)
    h = z_b
    b = np.arctan2(y_b - y, x_b - x)
    alpha_deg = np.random.uniform(ALPHA_MIN, ALPHA_MAX)
    alpha = np.deg2rad(alpha_deg)
    denom = 2 * (d * np.tan(alpha) - h) * np.cos(alpha) ** 2
    if denom <= 0:
        continue
    v = np.sqrt(G * d ** 2 / denom)
    if np.isnan(v) or np.isinf(v):
        continue
    # 加入噪声
    v_real = v + np.random.normal(0, V_STD)
    alpha_real = alpha + np.random.normal(0, ALPHA_STD)
    z_b_real = z_b + np.random.normal(0, ZB_STD)
    # 重新计算落点
    t = d / (v_real * np.cos(alpha_real))
    z_t = v_real * np.sin(alpha_real) * t - 0.5 * G * t ** 2
    z_landing = z_t
    # 命中判定
    if abs(z_landing - z_b_real) <= (RIM_RADIUS - BALL_RADIUS):
        rows.append([x, y, v_real, alpha_real, b])

columns = ['x', 'y', 'v', 'alpha', 'b']
df = pd.DataFrame(rows, columns=columns)
df.to_csv('basketball_dataset.csv', index=False)
print(f"生成数据集样本数: {len(df)}")
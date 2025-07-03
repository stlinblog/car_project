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
#空气阻力
K = 0.05  # 空气阻力系数（可调整，单位 1/s）

def landing_z_with_air(v0, x, y, alpha, b, k, d, t_guess):
    # 计算到篮筐的时间 t_guess
    # 反推 t_guess: d = (v0 * cos(alpha) / k) * (1 - exp(-k * t))
    # 解 t
    t = -np.log(1 - k * d / (v0 * np.cos(alpha))) / k
    # 计算 z(t)
    z = v0 * np.sin(alpha) * (1 - np.exp(-k * t)) / k - 0.5 * G * t ** 2
    return z, t

def solve_v_with_air(x, y, alpha, b, k, d, z_b, v_min=1, v_max=30, tol=1e-3):
    # 二分法求解合适的 v0
    for _ in range(100):
        v_mid = (v_min + v_max) / 2
        try:
            t = -np.log(1 - k * d / (v_mid * np.cos(alpha))) / k
        except:
            return None, None
        z = v_mid * np.sin(alpha) * (1 - np.exp(-k * t)) / k - 0.5 * G * t ** 2
        if abs(z - z_b) < tol:
            return v_mid, t
        if z > z_b:
            v_max = v_mid
        else:
            v_min = v_mid
    return None, None

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
    v, t = solve_v_with_air(x, y, alpha, b, K, d, z_b)
    if v is None:
        continue
    # 加入噪声
    v_real = v + np.random.normal(0, V_STD)
    alpha_real = alpha + np.random.normal(0, ALPHA_STD)
    z_b_real = z_b + np.random.normal(0, ZB_STD)
    # 重新计算落点
    t_real = -np.log(1 - K * d / (v_real * np.cos(alpha_real))) / K
    z_landing = v_real * np.sin(alpha_real) * (1 - np.exp(-K * t_real)) / K - 0.5 * G * t_real ** 2
    # 命中判定
    if abs(z_landing - z_b_real) <= (RIM_RADIUS - BALL_RADIUS):
        rows.append([x, y, v_real, alpha_real, b])

columns = ['x', 'y', 'v', 'alpha', 'b']
df = pd.DataFrame(rows, columns=columns)
df.to_csv('basketball_dataset.csv', index=False)
print(f"生成数据集样本数: {len(df)}")
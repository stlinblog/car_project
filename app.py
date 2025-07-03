import streamlit as st
import torch
import numpy as np
import joblib
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from train_model import Net
# 空气阻力系数
K = 0.05  # 单位 1/s，可根据实际情况调整
G = 9.8



@st.cache_resource
def load_model_and_scalers():
    model = Net()
    model.load_state_dict(torch.load('basketball_model.pth', map_location=torch.device('cpu')))
    model.eval()
    x_scaler = joblib.load('x_scaler.save')
    y_scaler = joblib.load('y_scaler.save')
    return model, x_scaler, y_scaler

def get_t_with_air(d, v, alpha, k):
    denom = v * np.cos(alpha)
    if denom == 0 or (k * d / denom) >= 1:
        return None
    t = -np.log(1 - k * d / denom) / k
    return t

def get_z_with_air(v, alpha, t, k, G=9.8):
    return v * np.sin(alpha) * (1 - np.exp(-k * t)) / k - 0.5 * G * t ** 2

def simulate_shots(N_TEST, only_hit=True):
    model, x_scaler, y_scaler = load_model_and_scalers()
    x_b, y_b, z_b = 10.0, 6.0, 2.0
    RIM_RADIUS = 0.23
    X_RANGE = (0, 5)
    Y_RANGE = (0, 5)
    hit_count = 0
    hit_params = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for _ in range(N_TEST):
        x = np.random.uniform(*X_RANGE)
        y = np.random.uniform(*Y_RANGE)
        X_input = np.array([[x, y]], dtype=np.float32)
        X_scaled = x_scaler.transform(X_input)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred = model(X_tensor).numpy()
        v, alpha, b = y_scaler.inverse_transform(pred)[0]
        d = np.sqrt((x_b - x) ** 2 + (y_b - y) ** 2)
        t = get_t_with_air(d, v, alpha, K)
        if t is None or np.isnan(t) or np.isinf(t):
            continue
        z_pred = get_z_with_air(v, alpha, t, K, G)
        is_hit = abs(z_pred - z_b) <= RIM_RADIUS
        if is_hit:
            hit_count += 1
            hit_params.append({
                "x": x,
                "y": y,
                "v": v,
                "alpha (deg)": np.rad2deg(alpha),
                "b (deg)": np.rad2deg(b)
            })
        if (only_hit and is_hit) or (not only_hit):
            t_vals = np.linspace(0, t, num=100)
            x_traj = x + np.cos(b) * v * np.cos(alpha) * (1 - np.exp(-K * t_vals)) / K
            y_traj = y + np.sin(b) * v * np.cos(alpha) * (1 - np.exp(-K * t_vals)) / K
            z_traj = get_z_with_air(v, alpha, t_vals, K, G)
            ax.plot(x_traj, y_traj, z_traj, alpha=0.6, color='b' if is_hit else 'gray')
    ax.scatter([x_b], [y_b], [z_b], color='red', s=100, label='Basket')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    return fig, hit_count, hit_params



st.title("篮球投篮命中率与轨迹模拟（已考虑空气阻力）")
mode = st.radio("选择模式", ["测试"])

if mode == "测试":
    N_TEST = st.slider("测试样本数", 1, 10000, step=1)
    only_hit = st.checkbox("只显示命中轨迹", value=True)
    if st.button("开始模拟"):
        with st.spinner("模拟中..."):
            fig, hit_count, hit_params = simulate_shots(N_TEST, only_hit)
            st.pyplot(fig)
            st.write(f"测试样本数: {N_TEST}")
            st.write(f"命中数: {hit_count}")
            st.write(f"命中率: {hit_count / N_TEST:.2%}")
            if hit_params:
                st.write("命中参数记录：")
                st.dataframe(hit_params)


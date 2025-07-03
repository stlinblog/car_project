# 篮球投篮轨迹与命中率模拟

本项目基于 PyTorch、Streamlit 和 matplotlib，实现了篮球投篮参数预测、命中率统计与三维轨迹可视化。用户可通过 Web 页面进行批量测试或单次测试，直观了解模型预测效果和投篮物理轨迹。

## 功能简介

- **批量测试**：随机生成多个投篮起点，统计命中率，并可视化所有或命中的投篮轨迹。
- **单次测试**：手动输入或随机生成投篮起点，展示该次投篮的三维轨迹、参数（初速度、仰角、方位角）及是否命中。
- **命中参数记录**：批量测试时，自动记录每次命中的参数，并以表格形式展示。

## 目录结构

```
car_project/
  ├── app.py                   # Streamlit Web 应用主程序
  ├── basketball_dataset.csv   # 训练数据集
  ├── basketball_model.pth     # 训练好的 PyTorch 模型
  ├── generate_dataset.py      # 数据集生成脚本
  ├── test_model_hit_rate.py   # 命中率与轨迹测试脚本（命令行版）
  ├── train_model.py           # 模型训练脚本
  ├── x_scaler.save            # 输入归一化器
  └── y_scaler.save            # 输出归一化器
```

## 环境依赖

- Python 3.7+
- torch
- numpy
- joblib
- matplotlib
- streamlit

安装依赖（推荐使用虚拟环境）：

```bash
pip install torch numpy joblib matplotlib streamlit
```

## 使用方法

### 1. 启动 Web 应用

在项目根目录下运行：

```bash
streamlit run app.py
```

浏览器会自动打开页面（如未自动打开，可访问 http://localhost:8501）。

### 2. 批量测试

- 选择"批量测试"模式。
- 设置测试样本数、是否只显示命中轨迹。
- 点击"开始模拟"，查看三维轨迹图、命中率统计和命中参数表格。

### 3. 单次测试

- 选择"单次测试"模式。
- 可手动输入起点 (x, y)，或点击"随机生成坐标"自动生成。
- 点击"单次测试"，查看该次投篮的三维轨迹、参数和命中结果。

### 4. 命令行测试（可选）

如需在命令行下批量测试，可运行：

```bash
python test_model_hit_rate.py
```

## 相关说明

- `basketball_model.pth`、`x_scaler.save`、`y_scaler.save` 需由 `train_model.py` 训练生成，或使用已提供的文件。
- 投篮物理参数和篮筐位置可在代码中调整。
- 本项目仅用于教学、科研或兴趣用途，模型精度依赖于数据和训练效果。

## 联系方式

如有问题或建议，欢迎 issue 或联系作者。 
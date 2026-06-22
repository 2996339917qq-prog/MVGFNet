import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Replace these with your real NSE values
# -----------------------------
models = ['DGCRN', 'GraphWave Net', 'STAEformer', 'DSTFformer']
datasets = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']

# Example: NSE values in the same order as models x datasets
nse_values = [
    [0.965, 0.961, 0.984, 0.9735],  # DGCRN
    [0.85, 0.959, 0.968, 0.974],  # GraphWave Net
    [0.9685, 0.9675, 0.986, 0.9745],  # STAEformer
    [0.9694, 0.9607, 0.9745, 0.9755]   # DSTFformer
]

# Build DataFrame
data = {'Model': [], 'Dataset': [], 'NSE': []}
for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        data['Model'].append(model)
        data['Dataset'].append(dataset)
        data['NSE'].append(nse_values[i][j])

df = pd.DataFrame(data)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(12, 8))

colors = ['#4ECDC4', '#45B7D1', '#FFA07A','#FF0000']
markers = ['o', 'o', 'o', 'o']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

for i, model in enumerate(models):
    model_data = df[df['Model'] == model]
    plt.scatter(
        model_data['Dataset'],
        model_data['NSE'],
        label=model,
        color=colors[i],
        marker=markers[i],
        s=200,
        edgecolors='black',
        linewidth=1.5,
        alpha=0.85
    )

plt.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.7)
plt.text(-0.3, 0.805, 'NSE = 0.8', color='gray', fontsize=12)

plt.title('Comparison of Nash-Sutcliffe Efficiency (NSE) Across Models', fontsize=18, fontweight='bold')
plt.xlabel('Dataset', fontsize=16, fontweight='bold')
plt.ylabel('NSE', fontsize=16, fontweight='bold')
plt.ylim(0.96, 1.0)  # zoom in from 0.96
plt.yticks(np.arange(0.96, 0.985, 0.005))  # ticks every 0.005

plt.xticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Model', fontsize=14, title_fontsize=16, loc='lower right', frameon=True, shadow=True)
plt.tight_layout()
plt.show()

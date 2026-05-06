import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

neutral = pd.read_csv('grid_cell_wind_speed_delta_neutral.csv', index_col=0, header=0)
stable = pd.read_csv('grid_cell_wind_speed_delta_stable.csv', index_col=0, header=0)
datasets = [neutral, stable]
labels = [r'N_WT $-$ N_NWT', r'S_WT $-$ S_NWT']
colors = ['blue', 'green']
markers = ['o', 'v']
linestyles = ['dashed', 'dotted']

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(2):
    ds = datasets[i]
    inv_deltax = ds['inv_deltax']
    mean_speeds = ds['actual']
    pred_delta_u = ds['predicted']
    ax.plot(inv_deltax, mean_speeds,
            color=colors[i], marker=markers[i],
            label=f'LES deficit: {labels[i]}',)
    ax.plot(inv_deltax, pred_delta_u, color=colors[i], linestyle=linestyles[i], linewidth=2.5, label=f'predicted in AIF: {labels[i]}')
ax.legend()
ax.set_xlabel(r'$1/\Delta x$ (1/m)')
ax.set_ylabel('Average upstream wind speed deficit (m/s)')
plt.tight_layout()
plt.savefig('mesoscale_cell_delta.png', bbox_inches='tight', dpi=300)
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

fig, axes = plt.subplots(ncols=2, figsize=(12, 5), width_ratios=[1, 1])

ax = axes[0]
for i in range(2):
    ds = datasets[i]
    inv_deltax = ds['inv_deltax']
    mean_speeds = ds['actual']
    pred_delta_u = ds['predicted']
    ax.plot(inv_deltax, mean_speeds,
            color=colors[i], marker=markers[i],
            label=f'LES deficit: {labels[i]}',)
    ax.plot(inv_deltax, pred_delta_u, color=colors[i], linestyle=linestyles[i], linewidth=2.5, label=f'predicted in AIF: {labels[i]}')
ax.plot([], [], color='turquoise', linestyle='dashdot', linewidth=2.5, label=f'predicted in AIF: both')
ax.set_xlabel(r'$1/\Delta x$ (1/m)')
ax.set_ylabel('Average upstream wind speed deficit (m/s)')

ax.legend(loc='center right')

ax = axes[1]
u_inftys = [9.567571, 9.740854]
for i in range(2):
    ds = datasets[i]
    u_infty = u_inftys[i]
    inv_deltax = ds['inv_deltax']
    mean_speeds = ds['actual'] / u_infty
    pred_delta_u = ds['predicted'] / u_infty
    ax.plot(inv_deltax, mean_speeds,
            color=colors[i], marker=markers[i],
            label=f'LES deficit: {labels[i]}',)
    if i == 1:
        ax.plot(inv_deltax, pred_delta_u, color='turquoise', linestyle='dashdot', linewidth=2.5, label=f'predicted in AIF: both')
ax.set_xlabel(r'$1/\Delta x$ (1/m)')
ax.set_ylabel('Average normalized upstream wind speed deficit')
plt.tight_layout()
plt.savefig('mesoscale_cell_delta.png', bbox_inches='tight', dpi=300)
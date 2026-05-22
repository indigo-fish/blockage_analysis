import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12

neutral = pd.read_csv('grid_cell_wind_speed_delta_neutral.csv', index_col=0, header=0)
stable = pd.read_csv('grid_cell_wind_speed_delta_stable.csv', index_col=0, header=0)
datasets = [neutral, stable]
labels = ['neutral', 'stable']
colors = ['blue', 'green']
markers = ['o', 'v']
linestyles = ['dashed', 'dotted']

u_inftys = [9.567571, 9.740854]

fig2, ax2 = plt.subplots(ncols=1, figsize=(6, 5))

i = 1
ds = datasets[i]
u_infty = u_inftys[i]
inv_deltax = ds['inv_deltax']
pred_delta_u = ds['predicted'] / u_infty

ax2.plot(inv_deltax, pred_delta_u, color='turquoise', linestyle='dashdot', linewidth=2.5, label=f'predicted in AIF')

ax2.set_xlabel(r'$1/\Delta x$ (1/m)')
ax2.set_ylabel('Average normalized upstream wind speed deficit')

ax2.set_title('Nondimensionalized wind speed deficit')

ax2.set_ylim(-.06, 0)

plt.tight_layout()
plt.savefig('hypothesis_mesoscale_cell_delta.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(ncols=1, figsize=(6, 5))

for i in range(2):
    ds = datasets[i]
    u_infty = u_inftys[i]
    inv_deltax = ds['inv_deltax']
    mean_speeds = ds['actual'] / u_infty
    pred_delta_u = ds['predicted'] / u_infty
    # calculate standard error from std and sample count (widths), propagate through normalization
    se = (ds['std'] / np.sqrt(ds['widths'])) / u_infty
    ax.errorbar(inv_deltax, mean_speeds,
                yerr=se,
                color=colors[i], marker=markers[i], linestyle='None', capsize=4,
                label=f'LES deficit: {labels[i]}')
    if i == 1:
        ax.plot(inv_deltax, pred_delta_u, color='turquoise', linestyle='dashdot', linewidth=2.5, label=f'predicted in AIF')
ax.set_xlabel(r'$1/\Delta x$ (1/m)')
ax.set_ylabel('Average normalized upstream wind speed deficit')

ax.set_title('Nondimensionalized wind speed deficit')
ax.legend(loc='lower left')

ax.set_ylim(-.06, 0)

plt.tight_layout()
plt.savefig('mesoscale_cell_delta.png', bbox_inches='tight', dpi=300)
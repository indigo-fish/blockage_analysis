import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

plt.rcParams['font.size'] = 14

# --- Secondary x-axis in rotor diameters ---
def meters_to_D(x):
    return x / 127

def D_to_meters(x):
    return x * 127

neutral_no_turbine = pd.read_csv('mean_vertical_slice_neutral_no_turbine.csv', index_col=0, header=0)
neutral_turbine = pd.read_csv('mean_vertical_slice_neutral_turbine.csv', index_col=0, header=0)
stable_no_turbine = pd.read_csv('mean_vertical_slice_stable_no_turbine.csv', index_col=0, header=0)
stable_turbine = pd.read_csv('mean_vertical_slice_stable_turbine.csv', index_col=0, header=0)
datasets = [[neutral_no_turbine, neutral_turbine],[stable_no_turbine, stable_turbine]]
labels = [['N_NWT', 'N_WT'], ['S_NWT', 'S_WT']]

fig = plt.figure(figsize=(12, 4))

outer = gridspec.GridSpec(ncols=2, nrows=1, hspace=0.2, wspace=0.1, width_ratios=[15, 1])

gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[0], wspace=0.2, hspace=0.3
)

axes = [[None, None], [None, None]]

panel_labels = [['a', 'b'], ['c', 'd']]

for i in range(2):
    for j in range(1, 2):
        if i == 0 and j == 0:
            ax = fig.add_subplot(gs[j - 1, i])
        else:
            ax = fig.add_subplot(gs[j - 1, i],
                                 sharex=axes[0][0],
                                 sharey=axes[0][0])

        axes[j][i] = ax
        ds = datasets[i][j]
        z = ds.index.astype('float').tolist()
        x = ds.columns.astype('float').tolist()

        levels = np.linspace(3.2, 11.6, 22)  # 20 intervals → 21 boundaries
        cf = ax.contourf(x, z, ds.values, levels=levels, cmap='viridis')

        if j == 1:
            ax.vlines([0], 90 - 127 / 2, 90 + 127 / 2,
                      color='black', linestyle='dashed', label='turbine position')
            ax.set_xlabel('X (m)')
            secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
            secax.tick_params(labeltop=False)
        else:
            ax.tick_params(labelbottom=False)
            secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
            secax.set_xlabel(r'X (Rotor Diameters, $D$)')
        if i == 0:
            ax.set_ylabel('height (m)')
        else:
            ax.tick_params(labelleft=False)

axes[1][0].legend(loc='upper right')

cax = fig.add_subplot(outer[1])
plt.colorbar(cf, cax=cax, label='Wind Speed (m/s)')

plt.tight_layout()
plt.savefig('vertical_slices.png', bbox_inches='tight', dpi=300)
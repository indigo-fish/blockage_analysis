import matplotlib
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 14

panel_labels = ['a', 'b']

unstable = pd.read_csv('time_evolution_unstable.csv', index_col=0, header=0).T
neutral = pd.read_csv('time_evolution_neutral.csv', index_col=0, header=0).T
stable = pd.read_csv('time_evolution_stable.csv', index_col=0, header=0).T

datasets = [unstable, neutral, stable]
labels = ['Unstable', 'Neutral', 'Stable']

levels = np.linspace(3, 12, 21)
cmap = plt.get_cmap('viridis')
norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(10, 10))
for i in range(3):
    ax = axes[i]
    ds = datasets[i]
    time = ds.columns.astype('datetime64[ns]').tolist()
    z = ds.index.astype(float).tolist()

    ax.contourf(time, z, ds, levels=levels, cmap=cmap, norm=norm)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()  # rotates dates nicely
    ax.set_xlabel('Time (hours)')

    ax.hlines([90], time[0], time[-1], color='black', linestyle='dashed', linewidth=2, label='hub height')
    ax.vlines(time[-3], 0, z[-1], color='orangered', linestyle='dotted', linewidth=3, label='inner domain start')
    ax.vlines(time[-2], 0, z[-1], color='mediumslateblue', linestyle='dashdot', linewidth=3, label='analysis start')

    ax.set_title(labels[i])

axes[0].legend(loc=(.7, 0.6))
plt.ylim(bottom=0)
axes[0].set_ylabel('height (m)')
axes[1].set_ylabel('height (m)')
axes[2].set_ylabel('height (m)')
axes[0].set_ylim(0, 300)
axes[1].set_ylim(0, 300)
axes[2].set_ylim(0, 300)

plt.tight_layout()
output_path = 'time_evolution.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)

cbar_fig, cbar_ax = plt.subplots(figsize=(1.5, 7))
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
colorbar_ticks = levels[::4]
cbar = cbar_fig.colorbar(sm, cax=cbar_ax, boundaries=levels, ticks=colorbar_ticks)
cbar.set_label('Wind speed (m/s)')
cbar_fig.tight_layout()
cbar_fig.savefig('time_evolution_colorbar.png', bbox_inches='tight', dpi=300)

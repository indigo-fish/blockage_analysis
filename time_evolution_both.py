import pandas as pd
import matplotlib.pyplot as plt

neutral = pd.read_csv('time_evolution_neutral.csv', index_col=0, header=0).T
stable = pd.read_csv('time_evolution_stable.csv', index_col=0, header=0).T
datasets = [neutral, stable]
labels = ['N_NWT', 'S_NWT']

fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 5), width_ratios=[12, 18])
for i in range(2):
    ax = axes[i]
    ds = datasets[i]
    time=ds.columns.astype('datetime64[ns]').tolist()
    z=ds.index.astype(float).tolist()
    cf = ax.contourf(time, z, ds, levels=20, cmap='viridis', vmin=3, vmax=12)


    plt.gcf().autofmt_xdate()  # rotates dates nicely
    ax.set_xlabel('Time')

    ax.hlines([90], time[0], time[-1], color='black', linestyle='dashed', linewidth=2, label='hub height')
    ax.vlines(time[-3], 0, z[-1], color='orangered', linestyle='dotted', linewidth=2, label='inner domain start')
    ax.vlines(time[-2], 0, z[-1], color='mediumslateblue', linestyle='dashdot', linewidth=2, label='analysis start')

    ax.set_title(labels[i])

plt.colorbar(cf, ax=ax, label='Wind Speed (m/s)')
ax.legend(loc='upper left')
plt.ylim(bottom=0)

plt.tight_layout()
output_path = 'time_evolution.png'
plt.savefig(output_path, dpi=200)
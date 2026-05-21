import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
panel_labels = ['a', 'b']

neutral = pd.read_csv('axial_wind_speed_neutral.csv', index_col=0, header=0)
stable = pd.read_csv('axial_wind_speed_stable.csv', index_col=0, header=0)
datasets = [neutral, stable]
labels = ['Neutral', 'Stable']
colors = ['blue', 'green']
linestyles = ['solid', 'dashed']

# --- Secondary axis ---
def meters_to_D(x):
    return x / 127


def D_to_meters(x):
    return x * 127


fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(6.5, 5))
for i in range(2):
    ds = datasets[i]
    mean = ds['mean']
    std = ds['std']
    x = ds['nx_dx']

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=colors[i],
        alpha=0.2,
        label=f'±1 Std Dev {labels[i]}'
    )

for i in range(2):
    ds = datasets[i]
    mean = ds['mean']
    std = ds['std']
    x = ds['nx_dx']

    ax.plot(
        x,
        mean,
        label=f'Mean {labels[i]}',
        color=colors[i],
        linestyle=linestyles[i],
    )
ax.set_xticks(np.linspace(-2000, 0, 5))
ax.set_xlabel('X (m)')

ax.set_xlim(-2000, 0)
ax.set_ylim(-2.5, 0.5)

secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
secax.set_xlabel(r'X (Rotor Diameters, $D$)')
secax.set_xticks(np.linspace(-16, 0, 5))

print("--- Upwind subset ---")
upwind_subset = ds.loc[ds['nx_dx'] < 0]
print(upwind_subset)
cutoff = min(upwind_subset['mean']) / 10
print(upwind_subset.loc[upwind_subset['mean'] == min(upwind_subset['mean'])])
print(upwind_subset.loc[upwind_subset['mean'] < cutoff])
print("Different from 0")
print(ds.loc[ds['mean'] + ds['std'] / np.sqrt(258) > 0])

ax.set_ylabel('Wind Speed (m/s)')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('axial_wind_speed_delta.png', bbox_inches='tight', dpi=300)
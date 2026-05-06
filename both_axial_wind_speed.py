import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
panel_labels = ['a', 'b']

neutral = pd.read_csv('axial_wind_speed_neutral.csv', index_col=0, header=0)
stable = pd.read_csv('axial_wind_speed_stable.csv', index_col=0, header=0)
datasets = [neutral, stable]
labels = [r'N_WT $-$ N_NWT', r'S_WT $-$ S_NWT']


# --- Secondary axis ---
def meters_to_D(x):
    return x / 127


def D_to_meters(x):
    return x * 127


fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 5), width_ratios=[1, 1])
for i in range(2):
    ax = axes[i]
    ds = datasets[i]
    mean = ds['mean']
    std = ds['std']
    x = ds['nx_dx']

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color='grey',
        alpha=0.5,
        label=f'±1 Std Dev'
    )

    ax.plot(
        x,
        mean,
        label=f'Mean',
        color='black'
    )

    ax.vlines([0], -4.5, 1,
              color='black', linestyle='dashed',
              label='turbine position')

    ax.set_xlabel('X (m)')

    ax.set_title(labels[i])

    secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
    secax.set_xlabel(r'X (Rotor Diameters, $D$)')

    print("--- Upwind subset ---")
    upwind_subset = ds.loc[ds['nx_dx'] < 0]
    print(upwind_subset)
    cutoff = min(upwind_subset['mean']) / 10
    print(upwind_subset.loc[upwind_subset['mean'] == min(upwind_subset['mean'])])
    print(upwind_subset.loc[upwind_subset['mean'] < cutoff])
    print("Different from 0")
    print(ds.loc[ds['mean'] + ds['std'] / np.sqrt(258) > 0])

    ax.text(
        0.02, 0.98, f'({panel_labels[i]})',
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold',
        va='top',
        ha='left'
    )
axes[0].set_ylabel('Wind Speed (m/s)')
axes[0].legend(loc='lower right')

plt.tight_layout()
plt.savefig('axial_wind_speed_delta.png', bbox_inches='tight', dpi=300)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIABLES = ['theta', 'windspeed', 'wind direction']
panel_labels = ['a', 'b', 'c']

UNIT_LS = [
    'potential temperature (K)',
    'wind speed (m s$^{-1}$)',
    'wind direction ($^\\circ$)'
]

XMIN_LS = [286, 0, 255]
XMAX_LS = [294, 12, 285]
NUM_X_TICKS_LS = [5, 4, 5]

LOCATION = f'vertical_profile_300_all_stabilities.png'

SCALING = 1
SHAPE = (5.3, 5.5)

color_map = {"stable": "#0038A8", "neutral": "#9B4F96", "unstable": "#D60270"}
detailed_label_map = {"stable": "stable (S_NWT)", "neutral": "neutral (N_NWT)", "unstable": "unstable (U10_z2)"}
linestyle_map = {"stable": "solid", "neutral": "dashed", "unstable": "dotted"}

fig, axes = plt.subplots(
    nrows=1, ncols=3,
    figsize=(16, 5.5),
    # sharey=True,
    constrained_layout=True
)

for i, VARIABLE in enumerate(VARIABLES):
    UNITS = UNIT_LS[i]
    XMIN, XMAX = XMIN_LS[i], XMAX_LS[i]
    num_x_ticks = NUM_X_TICKS_LS[i]
    ax = axes[i]


    stable = np.array(pd.read_csv(f'{VARIABLE}_vertical_profile_300_stable_000.csv', index_col=None, header=None))
    neutral = np.array(pd.read_csv(f'{VARIABLE}_vertical_profile_300_neutral_000.csv', index_col=None, header=None))
    all_data = {"stable": stable,
                "neutral": neutral,
                # "unstable": unstable
                }

    ax.hlines(90, xmin=XMIN, xmax=XMAX, color='dimgrey', linewidth=2)

    for temp_label, data in all_data.items():
        label = detailed_label_map[temp_label]
        color = color_map[temp_label]
        data[1, :] = data[1, :] / SCALING
        ax.plot(data[1, :], data[0, :], linewidth=3,
                 label=label,
                 color=color,
                 linestyle=linestyle_map[temp_label]
                 )

    ax.set_ylabel('height (m)', fontsize=24)
    ax.set_xlabel(UNITS, fontsize=24)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(0, 300)

    yticks = np.linspace(0, 300, 4)
    xticks = np.linspace(XMIN, XMAX, num_x_ticks)
    ax.set_yticks(yticks, labels=yticks.astype(int), fontsize=20)
    ax.set_xticks(xticks, labels=xticks.astype(int), fontsize=20)

    ax.axhspan(90 - 127/2, 90 + 127/2, facecolor='grey', alpha=0.5, label='turbine rotor area')
    if VARIABLE=='windspeed':
        ax.legend(fontsize=13, loc='center left')
    ax.text(
        0.02, 0.98, f'({panel_labels[i]})',
        transform=ax.transAxes,
        fontsize=24,
        fontweight='bold',
        va='top',
        ha='left'
    )

print(f'Saving to {LOCATION}')
plt.savefig(LOCATION, dpi=300, bbox_inches='tight')
plt.close()
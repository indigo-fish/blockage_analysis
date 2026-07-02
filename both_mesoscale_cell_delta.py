import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12

neutral = pd.read_csv('grid_cell_wind_speed_delta_neutral_std.csv', index_col=0, header=0)
stable = pd.read_csv('grid_cell_wind_speed_delta_stable_std.csv', index_col=0, header=0)
datasets = [neutral, stable]
labels = ['neutral', 'stable']
colors = ["#9B4F96", "#0038A8"]
colors_trend = ["#E5A8E0", "#5DA9FF"]
markers = ['o', 'v']
linestyles = ['dashed', 'dotted']

u_inftys = [9.567571, 9.740854]

fig2, ax2 = plt.subplots(ncols=1, figsize=(6, 5))


# store trendline equations
trend_equations = []

i = 1
ds = datasets[i]
u_infty = u_inftys[i]
inv_deltax = ds['inv_deltax']
pred_delta_u = ds['predicted'] / u_infty

ax2.plot(inv_deltax, pred_delta_u, color='turquoise', linestyle='dashdot', linewidth=2.5, label=f'predicted in AIF')

# -----------------------------
# LINEAR TREND LINE
# -----------------------------
coeffs = np.polyfit(inv_deltax, pred_delta_u, 1)
slope, intercept = coeffs

trend = np.poly1d(coeffs)

# smooth x values for plotting
x_fit = np.linspace(inv_deltax.min(), inv_deltax.max(), 200)

trend_equations.append(
    f"{'predicted'}: y = {slope:.4f}x"
)

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
                label=f'LES deficit: {labels[i]}', zorder=3)

    # -----------------------------
    # LINEAR TREND LINE
    # -----------------------------
    coeffs = np.polyfit(inv_deltax, mean_speeds, 1)
    slope, intercept = coeffs

    trend = np.poly1d(coeffs)

    # smooth x values for plotting
    x_fit = np.linspace(inv_deltax.min(), inv_deltax.max(), 200)

    trend_equations.append(
        f"{labels[i]}: y = {slope:.4f}x + {intercept:.4f}"
    )
    ax.plot(x_fit, trend(x_fit), color=colors_trend[i], linestyle='solid', linewidth=2.5, zorder=5)

    # annotate graph with the best fit lines in the appropriate colors
    ax.text(0.00088 + .000015 * i, -0.02 - i * 0.003, trend_equations[-1], color=colors[i])

    if i == 1:
        ax.plot(inv_deltax, pred_delta_u, color='turquoise', linestyle='dashdot', linewidth=2.5, label=f'predicted in AIF')
        # annotate graph with predicted line in the appropriate colors
        ax.text(0.00095, -0.02 - 2 * .003, trend_equations[0], color='lightseagreen')

# print equations
print("\nTrend line equations:")
for eq in trend_equations:
    print(eq)

print(-5.6697 / -32.8669)
print(-6.5410 / -32.8669)

ax.set_xlabel(r'$1/\Delta x$ (1/m)')
ax.set_ylabel('Average normalized upstream wind speed deficit')

ax.set_title('Nondimensionalized wind speed deficit')
ax.legend(loc='lower left')

ax.set_ylim(-.06, 0)

plt.tight_layout()
plt.savefig('mesoscale_cell_delta.png', bbox_inches='tight', dpi=300)
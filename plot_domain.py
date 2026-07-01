import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import rcParams

fig, ax = plt.subplots()

rcParams['font.size'] = 18

ax.tick_params(axis='both', which='major', labelsize=18)

rect = mpatches.Rectangle((.233 * 30, .166 * 30), 7, 5, ec='blue', fill=False, linewidth=2)
ax.add_patch(rect)

mesoscale_2 = mpatches.Rectangle((.233 * 30 + .5, .166 * 30 + 1.5), 2, 2, linestyle='dashed', linewidth=3, fill=True, facecolor='lightskyblue', ec='black')
mesoscale_2.set_facecolor((0.53, 0.81, 0.92, 0.4))
ax.add_patch(mesoscale_2)
mesoscale_1 = mpatches.Rectangle((.233 * 30 + 1.5, .166 * 30 + 2), 1, 1, linestyle='dotted', linewidth=3, fill=True, facecolor='rebeccapurple', ec='black')
mesoscale_1.set_facecolor((0.4, 0.2, 0.6, 0.4))
ax.add_patch(mesoscale_1)

plt.scatter([.233 * 30 + 2.5], [.166 * 30 + 2.5], color='blue', s=30, label='turbine location')


ax.set_xticks(np.arange(7, 15, 1))
ax.set_yticks(np.arange(5, 11, 1))

ax.set_xlabel('X (km)', fontsize=20)
ax.set_ylabel('Y (km)', fontsize=20)

fig.tight_layout()

plt.savefig('domain_full-cell.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots()

rcParams['font.size'] = 18

plt.scatter([.233 * 30 + 2.5], [.166 * 30 + 2.5], color='blue', s=30, label='turbine location')

ax.tick_params(axis='both', which='major', labelsize=18)

rect = mpatches.Rectangle((.233 * 30, .166 * 30), 7, 5, ec='blue', fill=False, linewidth=2)
ax.add_patch(rect)

mesoscale_1 = mpatches.Rectangle((.233 * 30 + 1.5, .166 * 30 + 2), 1, 1, linestyle='dotted', linewidth=3, fill=False, ec='gray')
ax.add_patch(mesoscale_1)
mesoscale_2 = mpatches.Rectangle((.233 * 30 + .5, .166 * 30 + 1.5), 2, 2, linestyle='dashed', linewidth=3, fill=False, ec='gray')
ax.add_patch(mesoscale_2)

mesoscale_4 = mpatches.Rectangle((.233 * 30 + 1.5, .166 * 30 + 1.5), 1, 2, linestyle='dashed', linewidth=3, fill=True, facecolor='lightskyblue', ec='black')
mesoscale_4.set_facecolor((0.53, 0.81, 0.92, 0.4))
ax.add_patch(mesoscale_4)
mesoscale_3 = mpatches.Rectangle((.233 * 30 + 2, .166 * 30 + 2), .5, 1, linestyle='dotted', linewidth=3, fill=True, facecolor='rebeccapurple', ec='black')
mesoscale_3.set_facecolor((0.4, 0.2, 0.6, 0.4))
ax.add_patch(mesoscale_3)

ax.set_xticks(np.arange(7, 15, 1))
ax.set_yticks(np.arange(5, 11, 1))

ax.set_xlabel('X (km)', fontsize=20)
ax.set_ylabel('Y (km)', fontsize=20)

fig.tight_layout()

plt.savefig('domain_half-cell.png', bbox_inches='tight', dpi=300)
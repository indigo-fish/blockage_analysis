import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import rcParams

fig, ax = plt.subplots()

rcParams['font.size'] = 18

plt.scatter([.233 * 30 + 2.5], [.166 * 30 + 2.5], color='blue', s=15, label='turbine location')

ax.tick_params(axis='both', which='major', labelsize=18)

rect = mpatches.Rectangle((.233 * 30, .166 * 30), 7, 5, ec='black', fill=False, linewidth=2)
ax.add_patch(rect)

plt.arrow(0.8, 12.5, 0.4, 0, width=.2, head_width=.6, color='grey')

ax.annotate("wind direction", (2.1, 12.2), fontsize=18)

ax.annotate("turbine", (9.7,7.2), fontsize=18)

ax.set_xticks(np.arange(0, 22, 3))
ax.set_yticks(np.arange(0, 16, 3))

ax.set_xlabel('X (km)', fontsize=20)
ax.set_ylabel('Y (km)', fontsize=20)

fig.tight_layout()

plt.savefig('domain.png', bbox_inches='tight', dpi=300)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl

# Set style parameters
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linewidth'] = 0.8
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['font.size'] = 16  # Increase base font size

# Define time points (x-axis values)
times = np.array([5, 30, 60, 300, 900, 3600, 14400])

# Define performance values for each model (y-axis values)
# TabPFN (PHE)
tabpfn_phe_values = np.array([0.98, 0.985, 0.99, 0.995, 0.99, 1.0, 0.99])
tabpfn_phe_std = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

# TabPFN
tabpfn_values = np.array([0.915, 0.915, 0.91, 0.93, 0.94, 0.975, 0.97])
tabpfn_std = np.array([0.02, 0.02, 0.02, 0.015, 0.015, 0.015, 0.015])

# AutoGluon
autogluon_values = np.array([0.92, 0.93, 0.93, 0.94, 0.95, 0.95, 0.95])
autogluon_std = np.array([0.02, 0.02, 0.02, 0.015, 0.015, 0.015, 0.015])

# CatBoost
catboost_values = np.array([0.84, 0.84, 0.84, 0.87, 0.89, 0.9, 0.9])
catboost_std = np.array([0.035, 0.035, 0.035, 0.03, 0.03, 0.025, 0.025])

# Create figure and axis
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot each model with appropriate styling - with white marker faces and colored edges
# TabPFN (PHE) - green pentagons
plt.plot(times, tabpfn_phe_values, 'g-', marker='p', markersize=15, 
         markerfacecolor='white', markeredgecolor='green', markeredgewidth=2,
         label='TabPFN (PHE)', linewidth=2.5)
plt.fill_between(times, tabpfn_phe_values - tabpfn_phe_std, tabpfn_phe_values + tabpfn_phe_std, color='green', alpha=0.2)

# TabPFN - blue stars
plt.plot(times, tabpfn_values, 'b-', marker='*', markersize=18, 
         markerfacecolor='white', markeredgecolor='blue', markeredgewidth=2,
         label='TabPFN', linewidth=2.5)
plt.fill_between(times, tabpfn_values - tabpfn_std, tabpfn_values + tabpfn_std, color='blue', alpha=0.2)

# AutoGluon - purple circles
plt.plot(times, autogluon_values, '-', color='purple', marker='o', markersize=15, 
         markerfacecolor='white', markeredgecolor='purple', markeredgewidth=2,
         label='AutoGluon', linewidth=2.5)
plt.fill_between(times, autogluon_values - autogluon_std, autogluon_values + autogluon_std, color='purple', alpha=0.2)

# CatBoost - red squares
plt.plot(times, catboost_values, 'r-', marker='s', markersize=14, 
         markerfacecolor='white', markeredgecolor='red', markeredgewidth=2,
         label='CatBoost', linewidth=2.5)
plt.fill_between(times, catboost_values - catboost_std, catboost_values + catboost_std, color='red', alpha=0.2)

# Add vertical dashed line at 14400 seconds
plt.axvline(x=14400, color='black', linestyle='--', linewidth=1.5)

# Set logarithmic scale for x-axis
plt.xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())

# Set the limits
plt.xlim(4, 16000)
plt.ylim(0.79, 1.01)

# Add grid
plt.grid(True, alpha=0.3)

# Label axes with larger fonts
plt.xlabel('Average Fit + Predict Time (s)', fontsize=18, fontweight='bold')
plt.ylabel('Normalized ROC AUC', fontsize=18, fontweight='bold')

# Customize legend with dotted lines connecting to markers
legend = plt.legend(loc='right', handlelength=2, fontsize=16)
for handle in legend.legendHandles:
    handle.set_linestyle(':')
    handle.set_linewidth(2.5)

# Set tick parameters with larger fonts
plt.tick_params(axis='both', which='major', labelsize=16)

# Add custom x-tick labels to match the figure
plt.xticks(times)

# Adjust layout
plt.tight_layout()

# Save the figure

plt.savefig(
        "results/publication_plots/nature_plots.png"
    )
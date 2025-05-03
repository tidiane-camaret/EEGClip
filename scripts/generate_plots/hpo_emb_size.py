import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
runs_df = pd.read_csv("results/hpo_emb_size.csv")

# Rename columns for consistent naming with our other plots
runs_df.rename(columns={
    "val_balanced_acc_logreg_gender": "accuracy_gender",
    "val_balanced_acc_logreg_medication": "accuracy_medication",
    "val_balanced_acc_logreg_pathological": "accuracy_pathological",
    "val_balanced_acc_logreg_under_50": "accuracy_age",
}, inplace=True)

# Set plot style
sns.set(style="whitegrid")
sns.set_context("poster")
plt.rcParams.update({'font.size': 16})

# Create figure
plt.figure(figsize=(20, 14))

# Get Set2 color palette with 4 colors
colors = sns.color_palette("Set2", 4)

# Create the plot with tasks in the specified order


# Plot each line with proper styling
for i, col in enumerate(['accuracy_pathological', 'accuracy_age', 'accuracy_gender', 'accuracy_medication']):
    # Get color and marker for this task
    color = colors[i]
    
    # Clean label (remove "accuracy_" prefix and capitalize)
    label = col.replace('accuracy_', '').capitalize()
    
    # Plot the line
    plt.plot(
        runs_df['projected_emb_dim'], 
        runs_df[col], 
        color=color, 
        linewidth=6,
        label=label
    )

# Set y-axis limits to focus on the relevant range
plt.ylim(0.5, 0.9)

# Customize plot appearance
#plt.title("Accuracy vs. Dimension of Projected Encodings", fontsize=32, fontweight='bold', pad=20)
plt.xlabel("Dimension of the shared embedding space", fontsize=50, fontweight='bold', labelpad=15)
plt.ylabel("Balanced Accuracy", fontsize=50, fontweight='bold', labelpad=15)

# Set tick parameters with larger font
plt.tick_params(axis='both', which='major', labelsize=50)

# Add grid
plt.grid(True, alpha=0.3, linewidth=1.5)

# Customize legend with dotted lines
legend = plt.legend(fontsize=35, title="Task", title_fontsize=40, 
                   frameon=True, framealpha=0.95, edgecolor='black',
                   loc='best')

# Make legend handles use dotted lines to match our earlier plots
for handle in legend.legendHandles:
    handle.set_linewidth(6)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig("results/publication_plots/hpo_emb_size.pdf", format="pdf", dpi=400, bbox_inches='tight')
plt.savefig("results/publication_plots/hpo_emb_size.png", format="png", dpi=400, bbox_inches='tight')

# Show plot
plt.show()
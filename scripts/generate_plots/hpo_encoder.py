import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
runs_df = pd.read_csv("results/hpo_lm.csv")

# Rename columns for clarity
runs_df.rename(columns={
    "val_balanced_acc_logreg_gender": "accuracy_gender",
    "val_balanced_acc_logreg_medication": "accuracy_medication",
    "val_balanced_acc_logreg_pathological": "accuracy_pathological",
    "val_balanced_acc_logreg_under_50": "accuracy_under_50",
}, inplace=True)

# Reshape the data for grouped bar plot
plot_data = runs_df.melt(
    id_vars=["text_encoder_name"], 
    value_vars=["accuracy_pathological","accuracy_under_50", "accuracy_gender", "accuracy_medication"  ],
    var_name="task", 
    value_name="accuracy"
)

# Clean up task names (remove "accuracy_" prefix)
plot_data["task"] = plot_data["task"].str.replace("accuracy_", "")
plot_data["task"] = plot_data["task"].str.capitalize()

# Set plot style with larger elements
plt.figure(figsize=(20, 14))
sns.set(style="whitegrid")
sns.set_context("poster")
plt.rcParams.update({'font.size': 16})

# Use a clearly distinct categorical color palette that differentiates the tasks
color_palette = sns.color_palette("Set2", 4)  # Categorical palette with distinct colors

# Create the grouped bar plot with larger bars
ax = sns.barplot(
    x="text_encoder_name", 
    y="accuracy", 
    hue="task", 
    data=plot_data,
    palette=color_palette,
    edgecolor='white',  # White edges for better separation
    linewidth=2,        # Thicker edges
    alpha=0.9,          # Slightly transparent for better readability
    errwidth=2,         # Error bar width
    capsize=0.1,        # Error cap size
)

# Set y-axis range from 0.5 to 1
plt.ylim(0.45, 0.9)

# Add grid lines on y-axis
plt.grid(axis='y', alpha=0.3, linewidth=1.5)

# Customize plot appearance
plt.title("Classification Accuracy by Pretrained Text Model", fontsize=32, fontweight='bold', pad=20)
plt.xlabel("Pretrained Text Model", fontsize=28, fontweight='bold', labelpad=15)
plt.ylabel("Balanced Accuracy", fontsize=28, fontweight='bold', labelpad=15)

# Rotate x-axis labels for readability and increase font size
plt.xticks(rotation=45, ha='right', fontsize=22)
plt.yticks(fontsize=22)

# Customize legend
legend = plt.legend(
    title="Task", 
    title_fontsize=26,
    fontsize=24, 
    loc='upper right', 
    frameon=True,
    framealpha=0.95,
    edgecolor='black'
)

# Tighten layout and ensure proper spacing for rotated labels
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Make room for the rotated x labels

# Save figure
plt.savefig("results/publication_plots/hpo_encoder.pdf", format="pdf", dpi=400, bbox_inches='tight')
plt.savefig("results/publication_plots/hpo_encoder.png", format="png", dpi=400, bbox_inches='tight')

# Show plot
plt.show()
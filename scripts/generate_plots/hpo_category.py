import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
runs_df = pd.read_csv("results/hpo_category_2.csv")

# Rename columns for clarity
runs_df.rename(columns={
    "val_balanced_acc_logreg_gender": "accuracy_gender",
    "val_balanced_acc_logreg_medication": "accuracy_medication",
    "val_balanced_acc_logreg_pathological": "accuracy_pathological",
    "val_balanced_acc_logreg_under_50": "accuracy_under_50",
}, inplace=True)

# Inverse row order
runs_df = runs_df.iloc[::-1]

# Define categories
cats = ["none", "all", "IMPRESSION", "DESCRIPTION OF THE RECORD", 
        "CLINICAL HISTORY", "MEDICATIONS", "INTRODUCTION", 
        "CLINICAL CORRELATION", "HEART RATE", "FINDINGS", "REASON FOR STUDY", 
        "TECHNICAL DIFFICULTIES", "EVENTS", "CONDITION OF THE RECORDING",
        "PAST MEDICAL HISTORY", "TYPE OF STUDY", "ACTIVATION PROCEDURES", "NOTE"]

runs_df["category"] = cats

# Sort dataframe by accuracy on pathological task (descending)
runs_df = runs_df.sort_values(by="accuracy_pathological", ascending=False)

# Reshape the data for grouped bar plot
plot_data = runs_df.melt(
    id_vars=["category"], 
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

# Use a distinct color palette different from the previous plot
# Previous plot used green/blue/purple/red spectrum, so use a different palette
color_palette = sns.color_palette("Set2", 4)  # A completely different color scheme

# Create the grouped bar plot with larger bars
ax = sns.barplot(
    x="category", 
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

# Set y-axis range from 0.5 to 0.9
plt.ylim(0.45, 0.8)

# Customize plot appearance
plt.title("Classification Accuracy by EEG Report Section", fontsize=32, fontweight='bold', pad=20)
plt.xlabel("Report Section", fontsize=28, fontweight='bold', labelpad=15)
plt.ylabel("Balanced Accuracy", fontsize=28, fontweight='bold', labelpad=15)

# Rotate x-axis labels for readability and increase font size
plt.xticks(rotation=45, ha='right', fontsize=22)
plt.yticks(fontsize=22)

# Adjust legend
plt.legend(
    title="Evaluation Task", 
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
plt.savefig("results/publication_plots/hpo_category.pdf", format="pdf", dpi=400, bbox_inches='tight')

# Show plot
plt.show()
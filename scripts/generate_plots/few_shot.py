import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FixedLocator
from sympy import Si

# Define task name as in your original script
task_name = "gender"

# Load your actual data
runs_df = pd.read_csv("results/classif_few_shot_test_set.csv")

# Only keep specified columns as in your original script
runs_df = runs_df[["freeze_encoder", "task_name", "weights", "val_acc_rec_balanced", "train_frac"]]
runs_df["accuracy"] = runs_df["val_acc_rec_balanced"]

# Process the weights column as in your original script
runs_df["weights"] = runs_df["weights"] + ["_trainable" if not x else "_frozen" for x in runs_df["freeze_encoder"]]

# Drop lines with freeze_encoder = True and weights = random at the same time
runs_df = runs_df[~(runs_df["weights"] == "random_frozen")]

# Remove occurrences when train_frac = 1
runs_df = runs_df[runs_df.train_frac != 1]

# Rename weights elements as in your original script
runs_df["weights"] = runs_df["weights"].replace(
    ["eegclip_frozen", "pathological_frozen", "under_50_frozen", "random_trainable"],
    ["EEG-Clip", "Irrelevant Task", "Irrelevant Task", "Task-Specific"]
)

# Set plot style with significantly larger font scaling
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linewidth'] = 0.8
sns.set_context("poster")  # Use 'poster' context for very large elements
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['font.size'] = 16  # Increase base font size

# Create larger figure
fig, ax = plt.subplots(1, 1, figsize=(20, 14))

# Filter data for specific task
task_data = runs_df[runs_df.task_name == task_name]

# ======= METHOD 1: TRANSFORM DATA FOR EVENLY SPACED X-AXIS =======
# Define the original data positions and their corresponding display positions
data_positions = [2, 5, 10, 20, 50]
display_positions = [0, 1, 2, 3, 4]  # Evenly spaced positions
fraction_labels = ["1/2", "1/5", "1/10", "1/20", "1/50"]  # Use strings for cleaner display

# Create a mapping between actual data points and display positions
position_map = dict(zip(data_positions, display_positions))

# Create a new column for the transformed x values
task_data = task_data.copy()
task_data['display_x'] = task_data['train_frac'].apply(lambda x: position_map.get(x, x))

# Create the plot with transformed x values
line_plot = sns.lineplot(
    data=task_data,
    x="display_x",  # Use the transformed positions
    y="accuracy",
    hue="weights",
    ax=ax,
    errorbar=('ci', 80),
    dashes=False,
    linewidth=6  # Much thicker lines
)

# Set the x-tick positions and labels
ax.set_xticks(display_positions)
ax.set_xticklabels(fraction_labels)

# Get the unique weight types to assign specific markers
weight_types = task_data['weights'].unique()
markers = {'EEG-Clip': 'p', 'Irrelevant Task': 'o', 'Task-Specific': 's'}

# Customize each line and its markers
for i, line in enumerate(ax.get_lines()):
    if i < len(weight_types):  # Only process the main lines, not confidence intervals
        # Get the color of the line
        color = line.get_color()
        
        # Set marker style based on weight type
        weight_type = weight_types[i % len(weight_types)]
        marker_style = markers.get(weight_type, 'o')  # Default to circle if not found
        
        # Apply marker settings
        line.set_marker(marker_style)
        line.set_markersize(55)  # Much larger markers
        line.set_markerfacecolor('white')  # White filled
        line.set_markeredgecolor(color)    # Colored outline
        line.set_markeredgewidth(3.5)      # Much thicker outline

# Customize axis labels and title with much larger fonts
ax.set_xlabel("Percentage of the training set", fontsize=50, fontweight='bold')
ax.set_ylabel("Balanced accuracy", fontsize=50, fontweight='bold')
task_name = "age" if task_name == "under_50" else task_name
ax.set_title(f"Evaluation Task : {task_name.capitalize()}", fontsize=50, fontweight='bold')

# Customize tick parameters with much larger font
ax.tick_params(axis='both', which='major', labelsize=55, length=10, width=2)

# Create and customize the legend
legend = plt.legend(loc='best', handlelength=3, fontsize=50, frameon=True, 
                    framealpha=0.95, edgecolor='black', borderpad=0.3,
                    labelspacing=0.5, handletextpad=1.0, ncol=1)

legend_texts = [text.get_text() for text in legend.get_texts()]

# Ensure legend shows correct markers with white fill and colored outlines
for i, handle in enumerate(legend.legendHandles):
    handle.set_linewidth(6)
    
    # Make sure marker is displayed in legend
    handle.set_markersize(55)  # Slightly smaller than in plot to save space
    
    # Get line color and apply it to marker edge
    color = handle.get_color()
    handle.set_markerfacecolor('white')
    handle.set_markeredgecolor(color)
    handle.set_markeredgewidth(3)
    
    # Get the label for this handle and set marker accordingly
    label = legend_texts[i]
    if label in markers:
        handle.set_marker(markers[label])
    else:
        # Default to circle if label not found in dictionary
        handle.set_marker('o')

# Set grid lines to only show for major ticks
ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.3)
ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.2)

# Turn off minor ticks
ax.minorticks_off()

# Save the figure
plt.tight_layout()
plt.savefig(f"results/publication_plots/few_shot_{task_name}.pdf", format="pdf", dpi=300, bbox_inches='tight')

# Display the plot (optional, comment out when running in batch mode)
plt.show()
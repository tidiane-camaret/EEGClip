import pandas as pd 
import wandb
api = wandb.Api()
import numpy as np

runs_df = pd.read_csv("results/classif_crossval.csv")

# print first 5 rows
print(runs_df.head())

# only keep "freeze_encoder,task_name,weights, val_acc_rec_balanced" columns
runs_df = runs_df[["freeze_encoder", "task_name", "weights", "val_acc_rec_balanced"]]

# weights = weights + ["_trainable" if x else "_frozen" for x in runs_df["freeze_encoder"]]

runs_df["weights"] = runs_df["weights"] + ["_trainable" if not x else "_frozen" for x in runs_df["freeze_encoder"]]

print(runs_df.head())

# for each task, plot the accuracy distribution for each weight


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, task in enumerate(runs_df.task_name.unique()):
    ax = axs[i//3, i%3]
    sns.boxplot(data=runs_df[runs_df.task_name==task], x="weights", y="val_acc_rec_balanced", ax=ax)
    ax.set_title(task)
    ax.set_ylim(0.5, np.max(runs_df[runs_df.task_name==task]["val_acc_rec_balanced"]))
    ax.set_xlabel("weights")
    ax.set_ylabel("accuracy")
    # all y values between 0.5 and 0.9
    ax.set_yticks(np.arange(0.45, 0.9, 0.05))


plt.show()
plt.savefig("results/classif_crossval.png")

# plot a table with mean and std for each task and each weight
print(runs_df.groupby(["task_name", "weights"]).agg(["mean", "std"]))


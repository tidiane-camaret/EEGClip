import pandas as pd 
import numpy as np

runs_df = pd.read_csv("results/classif_few_shot_test_set.csv")


# only keep "freeze_encoder,task_name,weights, val_acc_rec_balanced" columns
runs_df = runs_df[["freeze_encoder", "task_name", "weights", "val_acc_rec_balanced","train_frac"]]
runs_df["accuracy"] = runs_df["val_acc_rec_balanced"]

# weights = weights + ["_trainable" if x else "_frozen" for x in runs_df["freeze_encoder"]]

runs_df["weights"] = runs_df["weights"] + ["_trainable" if not x else "_frozen" for x in runs_df["freeze_encoder"]]

#print(runs_df.head())

# for each task, plot the accuracy distribution for each weight

#runs_df["weights"] = runs_df["weights"] + ["_trainable" if x else "_frozen" for x in runs_df["freeze_encoder"]]
# drop lines with freeze_encoder = True and weights = random at the same time
runs_df = runs_df[~(runs_df["weights"]=="random_frozen")]

# print row where freeze_encoder = True and weights = random
print(runs_df[(runs_df["freeze_encoder"]==True) & (runs_df["weights"]=="random")])

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# plot for each of the 6 tasks
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

for i, task in enumerate(["pathological","gender","under_50","medication"]):
    ax = axs[i//2, i%2]
    # plot accuracy vs train_frac for each weight
    sns.lineplot(data=runs_df[runs_df.task_name==task], x="train_frac", y="accuracy", hue="weights", ax=ax, )
    ax.set_title(task)
    ax.set_ylim(0.5, np.max(runs_df[runs_df.task_name==task]["accuracy"]))
    ax.set_xlabel("train_frac")
    ax.set_ylabel("accuracy")

plt.show()
# clear figure
plt.clf()

"""
fig, axs = plt.subplots(1, 2, figsize=(15, 10))

for i,task in enumerate(["epilep","seizure"]):
    ax = axs[i]
        # plot accuracy vs train_frac for each weight
    sns.lineplot(data=runs_df[runs_df.task_name==task], x="train_frac", y="accuracy", hue="weights", ax=ax, )
    ax.set_title(task)
    ax.set_ylim(0.5, 0.52)
    ax.set_xlabel("train_frac")
    ax.set_ylabel("accuracy")

plt.show()

"""
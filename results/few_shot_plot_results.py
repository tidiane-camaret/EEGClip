import pandas as pd 
import wandb
api = wandb.Api()
import numpy as np
"""
# Project is specified by <entity/project-name>
runs = api.runs("tidiane/EEGClip_few_shot")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("few_shot.csv")
"""
runs_df = pd.read_csv("few_shot.csv")

runs_df["accuracy"] = runs_df["summary"].apply(lambda x: eval(x)["val_acc_rec_balanced"])
for col in "freeze_encoder,task_name,weights,train_frac".split(","):
    runs_df[col] = runs_df["config"].apply(lambda x: eval(x)[col])

runs_df.drop(columns=["summary", "config", 'Unnamed: 0', 'name'], inplace=True)
print(runs_df.head())

#runs_df["weights"] = runs_df["weights"] + ["_trainable" if x else "_frozen" for x in runs_df["freeze_encoder"]]
# drop lines with freeze_encoder = True and weights = random at the same time
runs_df = runs_df[~((runs_df["freeze_encoder"]==True) & (runs_df["weights"]=="random"))]

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# plot for each of the 6 tasks
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, task in enumerate(runs_df.task_name.unique()):
    ax = axs[i//3, i%3]} Hello
    #drop lines where "weights" contains the task
    sns.lineplot(data=runs_df[runs_df.task_name==task][~runs_df.weights.str.contains(task)], x="train_frac", y="accuracy", hue="weights", ax=ax, )
    ax.set_title(task)
    ax.set_ylim(0.5, np.max(runs_df[runs_df.task_name==task]["accuracy"]))
    ax.set_xlabel("train_frac")
    ax.set_ylabel("accuracy")


plt.show()
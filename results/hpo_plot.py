"""
plot hpo results (val losses, accuracies)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


runs_df = pd.read_csv("results/hpo_lr.csv")

# keep lr_frac_lm, 'val_balanced_acc_logreg_gender', 'val_balanced_acc_logreg_medication',
#       'val_balanced_acc_logreg_pathological',
#       'val_balanced_acc_logreg_under_50'
# and rename them

runs_df.rename(columns={"val_balanced_acc_logreg_gender":"accuracy_gender",
                        "val_balanced_acc_logreg_medication":"accuracy_medication",
                        "val_balanced_acc_logreg_pathological":"accuracy_pathological",
                        "val_balanced_acc_logreg_under_50":"accuracy_under_50",}
                        , inplace=True)


sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10,6))
for col in ["accuracy_gender", "accuracy_medication","accuracy_pathological","accuracy_under_50"]:
    sns.lineplot(data=runs_df, x='lr_frac_lm', y=col, label=col, ax=ax)
    #ax.legend([col])
ax.set(xlabel='lr_frac_lm', ylabel='Accuracy')
ax.set_title('Impact of the learning rate of the language model on the accuracy for each task')
# log scale for x axis
ax.set_xscale('log')
# add legend for each curve

"""
ax2 = ax.twinx() 
ax2.set_ylabel('Loss') 
sns.lineplot(data=runs_df, x='lr_frac_lm', y='val_loss', ax=ax2, color='r')
ax2.legend(['val_loss'], loc='upper left')

"""
plt.tight_layout()
plt.show()
plt.savefig("results/hpo_lr.png")



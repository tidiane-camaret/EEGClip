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

runs_df.rename(columns={"val_balanced_acc_logreg_gender":"gender",
                        "val_balanced_acc_logreg_medication":"medication",
                        "val_balanced_acc_logreg_pathological":"pathological",
                        "val_balanced_acc_logreg_under_50":"age",}
                        , inplace=True)


sns.set(style="whitegrid")
sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(10,6))
for col in ["gender", "medication","pathological","age"]:
    sns.lineplot(data=runs_df, x='lr_frac_lm', y=col, label=col, ax=ax)
    #ax.legend([col])
ax.set_xlabel('LR of text encoder as a fraction of the general LR',fontsize=20)
ax.set_ylabel('Balanced accuracy',fontsize=20)
ax.set_title('Acc vs. learning rate of the text encoder for each task')
# log scale for x axis
ax.set_xscale('log')
# add legend for each curve
ax.tick_params(labelsize=20)
"""
ax2 = ax.twinx() 
ax2.set_ylabel('Loss') 
sns.lineplot(data=runs_df, x='lr_frac_lm', y='val_loss', ax=ax2, color='r')
ax2.legend(['val_loss'], loc='upper left')

"""
plt.tight_layout()
plt.show()
plt.savefig("results/publication_plots/hpo_lr_old.pdf", format="pdf", dpi=400, bbox_inches='tight')


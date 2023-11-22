"""
plot hpo results (val losses, accuracies)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


runs_df = pd.read_csv("results/hpo_emb_size.csv")


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
    sns.lineplot(data=runs_df, x='projected_emb_dim', y=col, label=col, ax=ax)
    #ax.legend([col])
ax.set_xlabel('Dimension of projected encodings',fontsize=20)
ax.set_ylabel('Balanced accuracy',fontsize=20)
ax.set_title('Acc vs. Dimension of projected encodings')

# add legend for each curve
ax.tick_params(labelsize=20)

plt.tight_layout()
plt.show()

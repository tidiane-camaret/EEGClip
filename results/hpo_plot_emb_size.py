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

runs_df.rename(columns={"val_balanced_acc_logreg_gender":"accuracy_gender",
                        "val_balanced_acc_logreg_medication":"accuracy_medication",
                        "val_balanced_acc_logreg_pathological":"accuracy_pathological",
                        "val_balanced_acc_logreg_under_50":"accuracy_under_50",}
                        , inplace=True)



# plot accuracies with projected_emb_dim as x axis
sns.lineplot(data=runs_df, x="projected_emb_dim",y="val_loss", label = "validation loss")

plt.xlabel("dimension of projected embeddings")
plt.ylabel("Validation loss")
plt.title("Validation loss for different embedding sizes")
plt.tight_layout()

plt.savefig("results/hpo_emb_size_loss.png")

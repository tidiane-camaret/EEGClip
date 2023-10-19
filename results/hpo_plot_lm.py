"""
plot hpo results (val losses, accuracies)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


runs_df = pd.read_csv("results/hpo_lm.csv")


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
sns.barplot(data=runs_df, x="text_encoder_name",y="val_loss", label = "validation loss",color='b')
# same color for all bars
plt.gca().set_prop_cycle(None)
# rotate x axis labels
plt.xticks(rotation=45)

plt.xlabel("text models")
plt.ylabel("Validation loss")
plt.title("Validation loss for different pretrained text models")
plt.tight_layout()

plt.savefig("results/hpo_lm_loss.png")

"""
# barplot of accuracies with text_encoder_name as x axis
runs_df.plot.bar(x="text_encoder_name", y=["accuracy_gender", "accuracy_medication", "accuracy_pathological", "accuracy_under_50"], rot=0)
                       
# rotate x axis labels
plt.xticks(rotation=45)

plt.xlabel("text models")
plt.ylabel("Accuracies")
plt.title("Accuracies for different pretrained text models")
plt.tight_layout()

# hide legend
plt.legend().set_visible(False)

plt.savefig("results/hpo_lm.png")
"""

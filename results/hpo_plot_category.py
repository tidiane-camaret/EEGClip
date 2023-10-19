"""
plot hpo results (val losses, accuracies)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


runs_df = pd.read_csv("results/hpo_category_2.csv")


# keep lr_frac_lm, 'val_balanced_acc_logreg_gender', 'val_balanced_acc_logreg_medication',
#       'val_balanced_acc_logreg_pathological',
#       'val_balanced_acc_logreg_under_50'
# and rename them

runs_df.rename(columns={"val_balanced_acc_logreg_gender":"accuracy_gender",
                        "val_balanced_acc_logreg_medication":"accuracy_medication",
                        "val_balanced_acc_logreg_pathological":"accuracy_pathological",
                        "val_balanced_acc_logreg_under_50":"accuracy_under_50",}
                        , inplace=True)

# inverse row order
runs_df = runs_df.iloc[::-1]

cats = ["none","all","IMPRESSION", "DESCRIPTION OF THE RECORD", \
                          "CLINICAL HISTORY", "MEDICATIONS", "INTRODUCTION", \
                          "CLINICAL CORRELATION", "HEART RATE", "FINDINGS", "REASON FOR STUDY", \
                          "TECHNICAL DIFFICULTIES", "EVENTS", "CONDITION OF THE RECORDING",\
                          "PAST MEDICAL HISTORY", "TYPE OF STUDY", "ACTIVATION PROCEDURES",\
                          "NOTE"]
print(len(cats))
runs_df["category"] = cats

# plot accuracies with category as x axis
sns.lineplot(data=runs_df, x="accuracy_gender",y="category", label = "accuracy_gender")
sns.lineplot(data=runs_df, x="accuracy_medication",y="category", label = "accuracy_medication")
sns.lineplot(data=runs_df, x="accuracy_pathological",y="category", label = "accuracy_pathological")
sns.lineplot(data=runs_df, x="accuracy_under_50",y="category", label = "accuracy_under_50")

plt.xlabel("Category")
plt.ylabel("Accuracy")
plt.title("Accuracies using different categories")
plt.tight_layout()

plt.savefig("results/hpo_category.png")

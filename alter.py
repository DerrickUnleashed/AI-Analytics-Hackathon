import pandas as pd
import numpy as np

df = pd.read_csv("CleanedDataset.csv")
df1 = df[df["isFraud"]==1]
df2 = df[df['isFraud'] == 0].sample(n=8213, random_state=42)
df_merged = pd.concat([df1, df2], ignore_index=True)
df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
df_merged = df_merged.drop(columns=['isSelfTransaction'])
df_merged.to_csv("BalancedDataset.csv",index=False)
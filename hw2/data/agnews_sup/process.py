import pandas as pd
import numpy as np

df = pd.read_csv('agnews_sup.csv', header=None)
# 删除title列
df = df.drop(columns=[1])
# 交换顺序
df = df[[2, 0]]
df.to_csv('agnews_sup_new.csv', index=False, header=True)
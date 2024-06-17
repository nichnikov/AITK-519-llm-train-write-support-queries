import os
import pandas as pd
import matplotlib.pyplot as plt 


df = pd.read_feather(os.path.join(os.getcwd(), "data_light", "queries_2020_0.feather"))
df["QueryLen"] = df["QueryText"].apply(lambda x: len(str(x).split()))
df["AnswerLen"] = df["Answer"].apply(lambda x: len(str(x).split()))
print(df)
print(df[["QueryText", "Answer", "QueryLen", "AnswerLen"]][:20])
print(df.info())

print(max(df["QueryLen"]))
print(max(df["AnswerLen"]))


df[:1000].plot.bar(rot=0)
# df["AnswerLen"].plot.bar()
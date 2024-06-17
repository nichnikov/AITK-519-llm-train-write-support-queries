import os, re
import pandas as pd
import matplotlib.pyplot as plt 


df = pd.read_feather(os.path.join(os.getcwd(), "data_light", "queries_2020_0.feather"))
df["QueryLen"] = df["QueryText"].apply(lambda x: len(str(x).split()))
df["AnswerLen"] = df["Answer"].apply(lambda x: len(str(x).split()))


"""
QueryLen Median = 48.0
QueryLen Mean = 60.662

AnswerLen Median = 134.0
AnswerLen Mean = 166.95275
"""

df_train = df[(df["QueryLen"] <= 100) & (df["AnswerLen"] <= 250)]
ptns = re.compile("\n|&nbsp")
print("patterns:", ptns)

#вопрос\nнаша
patterns = re.compile(r"\\n|\n|&nbsp;|\\t|\t|&laquo;|\s+")

df_train["Answer"] = df_train["Answer"].apply(lambda x: patterns.sub(" ", x))
df_train["Answer"] = df_train["Answer"].apply(lambda x: re.sub(r"\s+", " ", x))
df_train["QueryText"] = df_train["QueryText"].apply(lambda x: patterns.sub(" ", x))

print(df_train[["QueryText", "Answer"]][:10].to_dict(orient="records"))
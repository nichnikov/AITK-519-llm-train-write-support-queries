import os
import re
import pandas as pd

'''
fale_names = ["2024.xlsx"]
for fn in fale_names:
    df = pd.read_excel(os.path.join(os.getcwd(), "data", fn))
    df["QueryLen"] = df["QueryText"].apply(lambda x: len(str(x).split()))
    df["AnswerLen"] = df["Answer"].apply(lambda x: len(str(x).split()))
    out_fn = str(re.findall(r"\d+", fn)[0]) + ".feather"
    df.to_feather(os.path.join("data_with_length", out_fn))
    print(df)'''

df = pd.read_feather(os.path.join("data_with_length", "2024.feather"))
df_q15 = df[(df["QueryLen"] <= 20) & (df["QueryLen"] >= 5)]
print(df)
print(df_q15)
fn = "queries2024_short_exmpl1000.csv"
df_q15_1000 = df_q15[df_q15["Sys"] == "БСС"].sample(1000)
print(df_q15_1000)

patterns = "|".join(["&nbsp;", "&quot;", "&raquo;", "&mdash;", "&ndash;", "&laquo;",
                     "(?P<url>https?://[^\s]+)", "\n", "\s+"])

pattern = re.compile(patterns)
df_q15_1000["Answer"] = df_q15_1000["Answer"].apply(lambda x: pattern.sub(" ", x))
df_q15_1000["QueryText"] = df_q15_1000["QueryText"].apply(lambda x: pattern.sub(" ", x))
df_q15_1000.to_csv(os.path.join("data_with_length", fn), sep="\t", index=False)
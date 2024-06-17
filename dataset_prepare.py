import os, re
import pandas as pd
from os import listdir 
from os.path import isfile, join

def dataframe_handler(df: pd.DataFrame) -> dict:
    """
    QueryLen Median = 48.0
    QueryLen Mean = 60.662

    AnswerLen Median = 134.0
    AnswerLen Mean = 166.95275
    """

    df["QueryLen"] = df["QueryText"].apply(lambda x: len(str(x).split()))
    df["AnswerLen"] = df["Answer"].apply(lambda x: len(str(x).split()))
    df_train = df[(df["QueryLen"] <= 100) & (df["AnswerLen"] <= 250)]


    #вопрос\nнаша
    patterns = re.compile(r"\\n|\n|&nbsp;|\\t|\t|&laquo;|\s+")

    df_train["Answer"] = df_train["Answer"].apply(lambda x: patterns.sub(" ", x))
    df_train["Answer"] = df_train["Answer"].apply(lambda x: re.sub(r"\s+", " ", x))
    df_train["QueryText"] = df_train["QueryText"].apply(lambda x: patterns.sub(" ", x))
    df_train["label"] = "Правда"

    return df_train[["QueryText", "Answer", "label"]].to_dict(orient="records")


data_path = os.path.join(os.getcwd(), "data_light")
file_names = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for fn in file_names:
    df = pd.read_feather(os.path.join(os.getcwd(), "data_light", fn))
    fn_prefix = re.findall(r".*\.", fn)[0]
    print(fn_prefix)
    
    train_dicts = dataframe_handler(df)

    with open(os.path.join("datasets", fn_prefix + "txt"), 'w') as fout_t5:
        for d in train_dicts:
            fout_t5.write(f'Query: {d["QueryText"]} Document: {d["Answer"]} Relevant:\t{d["label"]}\n')


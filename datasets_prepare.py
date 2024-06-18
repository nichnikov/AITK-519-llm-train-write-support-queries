import os, re
from os import listdir 
from os.path import isfile, join
from random import shuffle
from math import sqrt
import pandas as pd

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
    patterns = re.compile(r"\\n|\n|&nbsp;|\\t|\t|&laquo;|&ndash;|\s+")

    df_train["Answer"] = df_train["Answer"].apply(lambda x: patterns.sub(" ", x))
    df_train["Answer"] = df_train["Answer"].apply(lambda x: re.sub(r"\s+", " ", x))
    df_train["QueryText"] = df_train["QueryText"].apply(lambda x: patterns.sub(" ", x))
    df_train["label"] = "Правда"

    """добавим отрицательные примеры:"""
    true_dicts = df_train[["QueryText", "Answer", "label"]].to_dict(orient="records")
    queries = [d["QueryText"] for d in true_dicts]
    answers = [d["Answer"] for d in true_dicts]
    queries_answers = list(zip(queries, answers))
    shuffle(queries_answers)
    queries, answers = zip(*queries_answers)
    # false_size = int(sqrt(len(queries)))
    false_size = 160
    false_dicts = [{"QueryText": queries[i], "Answer": answers[j], "label": "Ложь"} for i in 
                       range(false_size) for j in range(false_size) if i != j]
    train_dicts = true_dicts + false_dicts
    shuffle(train_dicts)
    return train_dicts


data_path = os.path.join(os.getcwd(), "data_light")
file_names = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for fn in file_names:
    print(fn)
    df = pd.read_feather(os.path.join(os.getcwd(), "data_light", fn))
    fn_prefix = re.findall(r".*\.", fn)[0]
    print(fn_prefix)
    
    all_dicts = dataframe_handler(df)
    train_size = int(len(all_dicts)*0.9)
    train_dicts = all_dicts[:train_size]
    val_dicts = all_dicts[train_size:]
    print("len train_dicts:", len(train_dicts))
    print("len val_dicts:", len(val_dicts))

    with open(os.path.join("datasets", "train", "train_" + fn_prefix + "txt"), 'w') as fout_t5_train:
        for d in train_dicts:
            fout_t5_train.write(f'Query: {d["QueryText"]} Document: {d["Answer"]} Relevant:\t{d["label"]}\n')

    with open(os.path.join("datasets", "val", "val_" + fn_prefix + "txt"), 'w') as fout_t5_val:
        for d in val_dicts:
            fout_t5_val.write(f'Query: {d["QueryText"]} Document: {d["Answer"]} Relevant:\t{d["label"]}\n')


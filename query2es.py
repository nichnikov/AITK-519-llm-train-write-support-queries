import os, re
import pandas as pd
import asyncio
from src.storage import ElasticClient
from src.texts_processing import TextsTokenizer

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def dict_handling(dd: dict) -> dict:
    pattern = re.compile(r"\n|&laquo;|&ndash;|&nbsp;|&raquo;|\s+")
    clear_answer = pattern.sub(" ", dd["Answer"])
    clear_query = pattern.sub(" ", dd["QueryText"])
    dd["ClearAnswer"] = clear_answer
    dd["ClearQuery"] = clear_query
    dd["AnswerUrls"] = re.findall("(?P<url>https?://[^\s]+)", clear_answer)
    return dd


if __name__ == "__main__":
    es = ElasticClient()
    tknz = TextsTokenizer()
    wr_index = "write_support_data"
    es.delete_index(wr_index)
    es.create_index(wr_index)
    
    fale_names = ["2020.xlsx", "2021.xlsx", "2022.xlsx", "2023.xlsx", "2024.xlsx"]
    # fale_names = ["2020.xlsx"]
    for fn in fale_names:
        year = re.findall(r"\d+", fn)[0]
        print("year:", year)
        df = pd.read_excel(os.path.join(os.getcwd(), "data", fn))
        df_dicts = df.to_dict(orient="records")
        data_dicts = [dict_handling(d) for d in df_dicts]
        print("start lematization")
        for d in data_dicts:
            d["LemQuery"] = " ".join(tknz([d["ClearQuery"]])[0])
            d["LemAnswer"] = " ".join(tknz([d["ClearAnswer"]])[0])
        print("end lematization")
        print(data_dicts[:10])
        es.add_docs(wr_index, data_dicts)

    
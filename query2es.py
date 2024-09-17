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
        data_dicts = [dict_handling(d) for d in df_dicts[:20]]
        print("start lematization")
        lem_queries = [" ".join(lm_q) for lm_q in tknz([d["ClearQuery"] for d in data_dicts])]
        lem_answers = [" ".join(lm_a) for lm_a in tknz([d["ClearAnswer"] for d in data_dicts])]
        for d, lq, la in zip(data_dicts, lem_queries, lem_answers):
            d["LemQuery"] = lq
            d["LemAnswer"] = la
        print("end lematization")
        print("start data sending")
        es.add_docs(wr_index, data_dicts)
        print("end data sending")

    
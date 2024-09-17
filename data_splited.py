import os, re
import pandas as pd

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


size = 20000
# fale_names = ["2020.xlsx", "2021.xlsx", "2022.xlsx", "2023.xlsx", "2024.xlsx"]
fale_names = ["2020.xlsx"]
for fn in fale_names:
    year = re.findall(r"\d+", fn)[0]
    print("year:", year)
    df = pd.read_excel(os.path.join(os.getcwd(), "data", fn))
    df_dicts = df.to_dict(orient="records")

    print(df_dicts[:10])
    pattern = re.compile(r"\n|&laquo;|&ndash;|&nbsp;|&raquo;|\s+")
    for d in df_dicts[:10]:
        print("Answer:\n", d["Answer"], "\n")

        # clear_answer = re.sub(r"\n|&laquo;|&ndash;|&nbsp;|&raquo;", " ", d["Answer"])
        # clear_answer = re.sub(r"\s+", " ", clear_answer)
        # d["clear_answer"] = re.sub(r"\s+", " ", clear_answer)
        clear_answer = pattern.sub(" ", d["Answer"])
        clear_query = pattern.sub(" ", d["QueryText"])
        d["clear_answer"] = clear_answer
        d["clear_query"] = clear_query

        d["urls"] = re.findall("(?P<url>https?://[^\s]+)", clear_answer)
        # print("clear answer:\n", clear_answer, "\n")
        # print("urls findall:", re.findall("(?P<url>https?://[^\s]+)", clear_answer))
        print(d, "\n")
        
    '''
    df_chunks = [df[i:i+size] for i in range(0,len(df), size)]
    for num, df_chunk in enumerate(df_chunks):
        print(year, num)
        nm_chunk = "_".join(["queries", str(year), str(num)])
        df_chunk.to_feather(os.path.join(os.getcwd(), "data_light", nm_chunk + ".feather"))
'''
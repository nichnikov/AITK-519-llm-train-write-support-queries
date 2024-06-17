import os, re
import pandas as pd

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


size = 20000
fale_names = ["2020.xlsx", "2021.xlsx", "2022.xlsx", "2023.xlsx", "2024.xlsx"]
for fn in fale_names:
    year = re.findall(r"\d+", fn)[0]
    print("year:", year)
    df = pd.read_excel(os.path.join(os.getcwd(), "data", fn))
    df_chunks = [df[i:i+size] for i in range(0,len(df), size)]
    for num, df_chunk in enumerate(df_chunks):
        print(year, num)
        nm_chunk = "_".join(["queries", str(year), str(num)])
        df_chunk.to_feather(os.path.join(os.getcwd(), "data_light", nm_chunk + ".feather"))

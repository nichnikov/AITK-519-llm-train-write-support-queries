import os
import pandas as pd

df = pd.read_excel(os.path.join(os.getcwd(), "data", "2020.xlsx"))
print(df)
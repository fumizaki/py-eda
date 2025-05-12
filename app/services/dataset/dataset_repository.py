import os
from glob import glob
import pandas as pd

class DatasetRepository:

    def __init__(self, dir: str):
        self.dir = dir
        self.options = [csv.split(f"{self.dir}/")[1] for csv in glob(os.path.join(f"{self.dir}", '*.csv'))]

    def get_dataframe(self, path: str) -> pd.DataFrame:
        return pd.read_csv(f"{self.dir}/{path}", encoding='utf_8')
    
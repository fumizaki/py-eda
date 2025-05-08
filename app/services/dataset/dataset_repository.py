import pandas as pd

class DatasetRepository:

    def __init__(self, dir: str):
        self.dir = dir
        self.options = [
            'Iris.csv',
            'Titanic.csv',
            'BostonHousing.csv',
            'Wine.csv',
        ]

    def get_dataframe(self, path: str) -> pd.DataFrame:
        return pd.read_csv(f"{self.dir}/{path}", encoding='utf_8')
    
import pandas as pd

def load_data(path):
    df = pd.read_json(path, lines=True)
    return df['headline'].values, df['is_sarcastic'].values
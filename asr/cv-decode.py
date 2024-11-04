import requests
import pandas as pd
from tqdm import tqdm

url = 'http://127.0.0.1:8001/asr'
data = pd.read_csv('data/cv-valid-dev.csv')
data['generated_text'] = pd.Series()

def transcribe(filename):
    with open(f'data/cv-valid-dev/{filename}', 'rb') as file:
        response = requests.post(url, files={'file': file})
    return response.json().get('transcription')

for i in tqdm(range(data.shape[0])):
    data.loc[i, 'generated_text'] = transcribe(data.loc[i, 'filename'])[0]

data_subset = pd.read_csv('cv-valid-dev.csv')
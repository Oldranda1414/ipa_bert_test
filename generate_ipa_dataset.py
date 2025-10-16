from phonemizer import phonemize
import pandas as pd
import os
from urllib.request import urlretrieve

def download(file, url):
    if not os.path.isfile(file):
        urlretrieve(url, file)

def phonemize_batch(batch):
    return phonemize(
        batch,
        language='en-us',
        backend='espeak',
        strip=False,
        preserve_punctuation=True
    )

def save_phonemized_data(ipa_data, filename):
    ipa_data.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data saved to {filename}")

# To reload the dataset:
def load_phonemized_data(filename):
    """Load phonemized data from CSV"""
    return pd.read_csv(filename, encoding='utf-8')

download('sst2_train.tsv', 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv')
df = pd.read_csv('sst2_train.tsv', delimiter='\t', header=None)

# Phonemize dataset
df[0] = phonemize_batch(df[0])
save_phonemized_data(df, "sst2_ipa_dataset/ipa_dataset.csv")
print("saved dataset")

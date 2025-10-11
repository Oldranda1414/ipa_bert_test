import os
import re
from urllib.request import urlretrieve
from phonemizer import phonemize
import pandas as pd

def download(file, url):
    if not os.path.isfile(file):
        urlretrieve(url, file)

def strip_tags(text):
    return text.replace("<br />", "\n")

def phonemize_batch(batch):
    # Step 1: split each text into parts (keeping track of structure)
    structured_parts = []
    all_text_parts = []  # flattened list to phonemize

    for text in batch:
        parts = re.split(r'(<br\s*/?>)', text, flags=re.IGNORECASE)
        current_structure = []
        for part in parts:
            if re.match(r'<br\s*/?>', part, flags=re.IGNORECASE) or part.strip() == "":
                # keep HTML and empty parts as-is
                current_structure.append((part, False))
            else:
                # mark normal text part for phonemization
                current_structure.append((len(all_text_parts), True))
                all_text_parts.append(part)
        structured_parts.append(current_structure)

    # Step 2: phonemize all text parts at once
    phonemized_all = phonemize(
        all_text_parts,
        language='en-us',
        backend='espeak',
        strip=False,
        preserve_punctuation=True
    )

    # Step 3: reconstruct the original texts efficiently
    result = []
    for structure in structured_parts:
        reconstructed = []
        for part, is_text in structure:
            if is_text:
                reconstructed.append(phonemized_all[part])  # index lookup
            else:
                reconstructed.append(part)
        result.append("".join(reconstructed))
    return result

def save_phonemized_data(ipa_data, filename):
    if not isinstance(ipa_data, pd.DataFrame):
        raise ValueError("ipa_data must be a pandas DataFrame")
    
    if not all(col in ipa_data.columns for col in ['text', 'label']):
        raise ValueError("DataFrame must contain 'text' and 'label' columns")
    
    ipa_data.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data saved to {filename}")

# To reload the dataset:
def load_phonemized_data(filename):
    """Load phonemized data from CSV"""
    return pd.read_csv(filename, encoding='utf-8')

download("imdb-train.csv.gz", "https://github.com/datascienceunibo/bbs-dl-lab-2019/raw/master/imdb-train.csv.gz")
train_set = pd.read_csv("imdb-train.csv.gz", sep="\t", names=["label", "text"])
download("imdb-test.csv.gz", "https://github.com/datascienceunibo/bbs-dl-lab-2019/raw/master/imdb-test.csv.gz")
test_set = pd.read_csv("imdb-test.csv.gz", sep="\t", names=["label", "text"])

# Phonemize dataset
train_set["text"] = phonemize_batch(train_set["text"])
save_phonemized_data(train_set, "ipa_dataset/ipa_train.csv")
print("saved train")
test_set["text"] = phonemize_batch(test_set["text"])
save_phonemized_data(test_set, "ipa_dataset/ipa_test.csv")
print("saved test")

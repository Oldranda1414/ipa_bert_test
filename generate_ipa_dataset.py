import re
import json
from phonemizer import phonemize
from datasets import load_dataset

imdb = load_dataset("imdb")

train_texts = imdb["train"]["text"]
train_labels = imdb["train"]["label"]

test_texts = imdb["test"]["text"]
test_labels = imdb["test"]["label"]

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
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ipa_data, f, ensure_ascii=False)

# Phonemize dataset
ipa_train = phonemize_batch(train_texts)
save_phonemized_data(ipa_train, "ipa_train.json")
print("saved train")
ipa_test = phonemize_batch(test_texts)
save_phonemized_data(ipa_test, "ipa_test.json")
print("saved test")

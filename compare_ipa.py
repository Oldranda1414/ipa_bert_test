import pandas as pd
from transformers import AutoTokenizer

def strip_tags(text):
    return text.replace("<br />", "\n")

def load_phonemized_data(filename):
    return pd.read_csv(filename, encoding='utf-8')

def prepare_ipa_text(text):
    # Split words by space (if words are already separated)
    words = text.strip().split()

    processed_words = []
    for word in words:
        # Split into individual phonemes (characters)
        phonemes = list(word)
        # Join with space and insert WORD_BOUNDARY between words
        processed_words.append(" ".join(phonemes))

    # Join all words with WORD_BOUNDARY markers
    return " WORD_BOUNDARY ".join(processed_words)

def main() -> None:
    ipa_train_set = load_phonemized_data("./imdb_ipa_dataset/ipa_train.csv")
    ipa_train_set["text"] = ipa_train_set["text"].apply(strip_tags)

    ipa_tokenizer_name = {"pretrained_model_name_or_path":'phonemetransformers/ipa-childes-tokenizers', "subfolder":'EnglishUK'}
    tokenizer = AutoTokenizer.from_pretrained(**ipa_tokenizer_name)

    sample_texts = ipa_train_set["text"].apply(prepare_ipa_text)[:5].to_list()
    enc = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

    print("Tokens:", enc["input_ids"])
    print("Decoded back:", [tokenizer.decode(ids) for ids in enc["input_ids"]])

main()

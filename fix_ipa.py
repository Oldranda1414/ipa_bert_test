import pandas as pd
from transformers import AutoTokenizer
import re

def strip_tags(text):
    return text.replace("<br />", "\n")

def load_phonemized_data(filename):
    return pd.read_csv(filename, encoding='utf-8')

def normalize_to_tokenizer_vocabulary(text):
    """
    Convert text to only use characters in the tokenizer's vocabulary
    """
    # Remove all punctuation (since tokenizer doesn't support it)
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', text)
    
    # Convert to lowercase (since tokenizer only has lowercase)
    text = text.lower()
    
    # Replace multiple spaces with single spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def prepare_ipa_text_for_tokenizer(text):
    """
    Prepare IPA text specifically for this tokenizer
    """
    # First normalize
    text = normalize_to_tokenizer_vocabulary(text)
    
    # Split into words
    words = text.strip().split()
    
    processed_words = []
    for word in words:
        # The tokenizer expects whole IPA words, not space-separated phonemes
        # So we keep words together but mark boundaries
        processed_words.append(word)
    
    # Join with WORD_BOUNDARY (which is in the tokenizer's vocabulary)
    return " WORD_BOUNDARY ".join(processed_words)

def analyze_tokenizer_expectations(tokenizer):
    """Understand exactly what the tokenizer expects"""
    
    print("=== WHAT THIS TOKENIZER EXPECTS ===")
    print("This tokenizer is designed for UK English IPA transcriptions")
    print("It expects complete IPA words, not individual phonemes")
    print("\nSupported IPA sequences include:")
    
    vocab = tokenizer.get_vocab()
    ipa_sequences = [token for token in vocab.keys() if len(token) > 1 and token not in ['UNK', 'PAD', 'WORD_BOUNDARY', 'UTT_BOUNDARY']]
    
    # Group by type
    diphthongs = [seq for seq in ipa_sequences if any(seq.endswith(x) for x in ['ɪ', 'ʊ', 'ə'])]
    long_vowels = [seq for seq in ipa_sequences if 'ː' in seq]
    consonants = [seq for seq in ipa_sequences if seq not in diphthongs and seq not in long_vowels and len(seq) > 1]
    
    print(f"Diphthongs: {diphthongs}")
    print(f"Long vowels: {long_vowels}") 
    print(f"Consonant clusters: {consonants}")
    
    return ipa_sequences

def test_correct_usage(tokenizer):
    """Test how to properly use this tokenizer"""
    
    print("\n=== PROPER USAGE EXAMPLES ===")
    
    # These should work well with the tokenizer
    test_examples = [
        "hɛləʊ WORD_BOUNDARY wɜːld",  # "hello world" in UK IPA
        "ɡʊd WORD_BOUNDARY mɔːnɪŋ",   # "good morning" 
        "θæŋk WORD_BOUNDARY jʊ",      # "thank you"
        "əʊ WORD_BOUNDARY kɛɪ",       # "okay"
    ]
    
    for example in test_examples:
        tokens = tokenizer.encode(example, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        print(f"Input:  '{example}'")
        print(f"Tokens: {tokens}")
        print(f"Output: '{decoded}'")
        print()

def main() -> None:
    ipa_train_set = load_phonemized_data("./imdb_ipa_dataset/ipa_train.csv")
    ipa_train_set["text"] = ipa_train_set["text"].apply(strip_tags)

    ipa_tokenizer_name = {"pretrained_model_name_or_path":'phonemetransformers/ipa-childes-tokenizers', "subfolder":'EnglishUK'}
    tokenizer = AutoTokenizer.from_pretrained(**ipa_tokenizer_name)
    
    # Understand the tokenizer
    analyze_tokenizer_expectations(tokenizer)
    test_correct_usage(tokenizer)
    
    # Process your data correctly
    print("=== PROCESSING YOUR DATA ===")
    sample_texts = ipa_train_set["text"][:5].to_list()
    
    for i, original_text in enumerate(sample_texts):
        print(f"\n--- Sample {i} ---")
        print(f"Original: {original_text[:100]}...")
        
        # Normalize and prepare for tokenizer
        normalized = normalize_to_tokenizer_vocabulary(original_text)
        prepared = prepare_ipa_text_for_tokenizer(normalized)
        
        print(f"Normalized: {normalized[:100]}...")
        print(f"Prepared: {prepared[:100]}...")
        
        # Tokenize
        enc = tokenizer(prepared, padding=True, truncation=True, return_tensors="pt")
        decoded = tokenizer.decode(enc["input_ids"][0])
        
        print(f"Tokenized length: {len(enc['input_ids'][0])}")
        print(f"Decoded: {decoded}")
        
        # Check for UNK tokens
        unk_count = sum(1 for token_id in enc["input_ids"][0] if token_id == tokenizer.unk_token_id)
        print(f"UNK tokens: {unk_count}")

if __name__ == "__main__":
    main()

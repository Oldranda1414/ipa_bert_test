import pandas as pd
from transformers import AutoTokenizer
from collections import Counter
import re

def strip_tags(text):
    return text.replace("<br />", "\n")

def load_phonemized_data(filename):
    return pd.read_csv(filename, encoding='utf-8')

def prepare_ipa_text(text, known_phonemes=None):
    """
    Split the IPA text into phoneme tokens based on the tokenizer's vocabulary
    rather than individual characters.
    """
    if known_phonemes is None:
        raise ValueError("Must pass known_phonemes from tokenizer vocab")

    # Remove all punctuation (since tokenizer doesn't support it)
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', text)
    
    # Convert to lowercase (since tokenizer only has lowercase)
    text = text.lower()
    
    # Replace multiple spaces with single spaces
    text = re.sub(r'\s+', ' ', text)

    # Sort phonemes by descending length so that longer matches (like 'tʰ' or 'ɑː') are matched first
    sorted_phonemes = sorted(known_phonemes, key=len, reverse=True)

    # Build regex to match any known phoneme
    phoneme_pattern = re.compile("|".join(map(re.escape, sorted_phonemes)))

    processed_sentences = []
    for word in text.strip().split():
        # Find all phonemes in this word
        phonemes = phoneme_pattern.findall(word)
        if not phonemes:
            # fallback: split into characters if nothing matches (rare)
            phonemes = list(word)
        processed_sentences.append(" ".join(phonemes))

    return " WORD_BOUNDARY ".join(processed_sentences)

def analyze_unknown_tokens(original_texts, tokenizer, tokenized_output):
    """Analyze which characters are being tokenized as unknown tokens"""
    
    # Get the unknown token ID
    unk_token_id = tokenizer.unk_token_id
    print(f"Unknown token ID: {unk_token_id}")
    
    unknown_characters = Counter()
    unknown_positions = []
    
    for i, (original_text, token_ids) in enumerate(zip(original_texts, tokenized_output["input_ids"])):
        # Decode tokens back to text
        decoded_tokens = [tokenizer.decode(token_id) for token_id in token_ids]
        
        # Find positions of unknown tokens
        unk_positions = [j for j, token_id in enumerate(token_ids) if token_id == unk_token_id]
        
        if unk_positions:
            print(f"\n--- Sample {i} ---")
            print(f"Original: {original_text}")
            print(f"Decoded: {tokenizer.decode(token_ids)}")
            print(f"Unknown token positions: {unk_positions}")
            
            # Analyze the original text around unknown token positions
            for pos in unk_positions:
                # Get the token that was unknown
                token_text = decoded_tokens[pos]
                
                # Try to find which part of the original text corresponds to this position
                # This is approximate since tokenization changes the length
                print(f"  Position {pos}: '{token_text}'")
                
                # Add to our counter
                unknown_characters[token_text] += 1
                unknown_positions.append({
                    'sample_idx': i,
                    'token_pos': pos,
                    'token_text': token_text,
                    'original_text_snippet': original_text[max(0, pos-10):min(len(original_text), pos+10)]
                })
    
    return unknown_characters, unknown_positions

def detailed_character_analysis(original_texts, tokenizer):
    """Analyze how individual characters are tokenized"""
    
    # Get all unique characters from the texts
    all_chars = set()
    for text in original_texts:
        all_chars.update(text)
    
    print("\n=== CHARACTER-BY-CHARACTER ANALYSIS ===")
    char_analysis = []
    
    for char in sorted(all_chars):
        # Tokenize just this single character
        tokens = tokenizer.encode(char, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        
        char_info = {
            'char': char,
            'char_repr': repr(char),
            'token_ids': tokens,
            'decoded': decoded,
            'is_unknown': any(token_id == tokenizer.unk_token_id for token_id in tokens),
            'num_tokens': len(tokens)
        }
        char_analysis.append(char_info)
        
        if char_info['is_unknown'] or len(tokens) > 1:
            print(f"Char: {char!r:10} -> Tokens: {tokens} -> Decoded: {decoded!r}")
    
    return char_analysis

def analysis() -> None:
    ipa_train_set = load_phonemized_data("./imdb_ipa_dataset/ipa_train.csv")
    ipa_train_set["text"] = ipa_train_set["text"].apply(strip_tags)

    ipa_tokenizer_name = {"pretrained_model_name_or_path":'phonemetransformers/ipa-childes-tokenizers', "subfolder":'EnglishUK'}
    tokenizer = AutoTokenizer.from_pretrained(**ipa_tokenizer_name)
    
    print(f"Tokenizer info:")
    print(f"  Unknown token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Get phoneme vocabulary (excluding special tokens)
    known_phonemes = [p for p in tokenizer.get_vocab().keys() if p not in {"WORD_BOUNDARY", "UTT_BOUNDARY", "PAD", "UNK"}]

    # Apply improved IPA preparation
    sample_texts = ipa_train_set["text"].apply(lambda x: prepare_ipa_text(x, known_phonemes))[:5].to_list()
    
    print("\n=== ORIGINAL SAMPLE TEXTS ===")
    for i, text in enumerate(sample_texts):
        print(f"Sample {i}: {text[:100]}...")  # First 100 chars
    
    # Tokenize
    enc = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")
    
    print("\n=== TOKENIZATION RESULTS ===")
    print("Tokens:", enc["input_ids"])
    print("Decoded back:", [tokenizer.decode(ids) for ids in enc["input_ids"]])
    
    # Analyze unknown tokens
    unknown_chars, unknown_details = analyze_unknown_tokens(sample_texts, tokenizer, enc)
    
    print("\n=== UNKNOWN CHARACTERS SUMMARY ===")
    for char, count in unknown_chars.most_common():
        print(f"'{char}' (repr: {repr(char)}): {count} occurrences")
    
    # Detailed character analysis
    char_analysis = detailed_character_analysis(sample_texts, tokenizer)
    
    # Save analysis results
    analysis_df = pd.DataFrame(char_analysis)
    analysis_df.to_csv("character_tokenization_analysis.csv", index=False, encoding='utf-8')
    
    unknown_details_df = pd.DataFrame(unknown_details)
    unknown_details_df.to_csv("unknown_tokens_details.csv", index=False, encoding='utf-8')
    
    print(f"\nAnalysis saved to files:")
    print(f"  - character_tokenization_analysis.csv")
    print(f"  - unknown_tokens_details.csv")

def print_vocab():
    ipa_tokenizer_name = {"pretrained_model_name_or_path":'phonemetransformers/ipa-childes-tokenizers', "subfolder":'EnglishUK'}
    tokenizer = AutoTokenizer.from_pretrained(**ipa_tokenizer_name)
    print([item[0] for item in tokenizer.get_vocab().items()])

def main() -> None:
    print_vocab()
    analysis()

if __name__ == "__main__":
    main()

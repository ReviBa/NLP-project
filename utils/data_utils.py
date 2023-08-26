from transformers import T5Tokenizer
import string


def get_tokenizer_based_on_data(df, base_tokenizer, min_frequency=3):
    tokenizer = T5Tokenizer.from_pretrained(base_tokenizer, model_max_length=512)
    all_tokens = []

    for idx, row in df.iterrows():
        line = "{speaker}: {line}".format(speaker=row['speaker'], line=row['line'])
        tokenized = ['_' + word.strip(string.punctuation) for word in line.split() if
                     word.strip(string.punctuation).isalnum()]
        all_tokens += tokenized
    tokens_freq = dict()

    for token in all_tokens:
        if token in tokens_freq:
            tokens_freq[token] += 1
        else:
            tokens_freq[token] = 1

    filtered_tokens_freq = {key: value for key, value in tokens_freq.items() if value >= min_frequency}
    all_tokens = list(filtered_tokens_freq.keys())
    all_tokens = set(all_tokens)  # remove duplications
    new_tokens = all_tokens - set(tokenizer.get_vocab().keys())
    tokenizer.add_tokens(list(new_tokens))
    return tokenizer


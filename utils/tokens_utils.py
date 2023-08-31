from transformers import T5Tokenizer
import string


def get_tokenizer_based_on_data(df, speaker_col_name, line_col_name, base_tokenizer="t5-base", min_frequency=3):
    tokenizer = T5Tokenizer.from_pretrained(base_tokenizer, model_max_length=max(get_max_tokens_number(df, speaker_col_name, line_col_name)))
    all_tokens = []

    for idx, row in df.iterrows():
        line = "{speaker}: {line}".format(speaker=row[speaker_col_name], line=row[line_col_name])
        # underscore is added to each token to fit the format of SentencePiece tokenizer
        # we tried to use SentencePiece itself, but it completely broke the tokenization
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


def get_max_tokens_number(df, col_name1, col_name2):
    wc_q = df[col_name1].apply(lambda x: len(str(x).split()))
    wc_a = df[col_name2].apply(lambda x: len(str(x).split()))
    input_max_len = wc_q.max()
    output_max_len = wc_a.max()
    return input_max_len, output_max_len

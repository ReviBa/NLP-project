import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def padding(text_arr, max_length, pad_token):
    padding_size = max_length - len(text_arr)
    return text_arr + padding_size * [pad_token]


def encode_sentence(sentence, tokenizer):
    tokenized = tokenizer.tokenize(sentence)
    sent_to_id = tokenizer.convert_tokens_to_ids(tokenized)
    if sent_to_id == []:
        print(sentence)
        print(tokenized)
    encoded = tokenizer.encode(sent_to_id)
    return encoded


def prepare_data(data, label2id, tokenizer, batch_size=8, pad_token=0):
    prepared_sentences = []
    encoded_sentences = []
    prepared_labels = []
    attention_masks = []

    for index, row in data.iterrows():
        sentence, label = row['line'], row['speaker']
        encoded_sent = encode_sentence(sentence, tokenizer)
        encoded_sentences.append(encoded_sent)
        prepared_labels.append(label2id[label])

    max_seq_length = max([len(sen) for sen in encoded_sentences])

    for encoded_sent in encoded_sentences:
        prepared_sent = padding(encoded_sent, max_seq_length, pad_token)
        prepared_sentences.append(prepared_sent)
        attention_masks.append([int(token_id > 0) for token_id in prepared_sent])

    sent_tensor = torch.tensor(prepared_sentences)
    labels_tensor = torch.tensor(prepared_labels)
    attent_tensor = torch.tensor(attention_masks)

    dataset = TensorDataset(sent_tensor, attent_tensor, labels_tensor)
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

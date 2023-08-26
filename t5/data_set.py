class T5DataSet:

    def __init__(self, input, target, tokenizer, input_max_len, output_max_len):
        self.input = input
        self.target = target
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        question = str(self.input[item])
        answer = str(self.target[item])

        input_tokenize = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.input_max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        output_tokenize = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=self.output_max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"

        )
        input_ids = input_tokenize["input_ids"].flatten()
        attention_mask = input_tokenize["attention_mask"].flatten()
        labels = output_tokenize['input_ids'].flatten()
        # in case we will want to train ligning module again return out as dictionary:
        out = {
            'question': question,
            'answer': answer,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': labels
        }

        return out

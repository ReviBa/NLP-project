import re


def clean_response(response):
    response = re.sub('<[^>]+>', '', response)
    response = response.strip()
    return response


class DialogContextManager:
    def __init__(self, model, tokenizer, max_input_len):
        self.context = []
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.speaker_str = 'Person1'

    def write(self, message):
        self.context.append(f"{self.speaker_str}:{message}")

        inputs_encoding = self.tokenizer(
            "".join(self.context),
            add_special_tokens=True,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        generate_ids = self.model.model.generate(
            input_ids=inputs_encoding["input_ids"],
            attention_mask=inputs_encoding["attention_mask"],
            max_length=self.max_input_len,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        preds = [
            self.tokenizer.decode(gen_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            for gen_id in generate_ids
        ]

        response = clean_response("".join(preds))
        self.context.append(response)

        return response

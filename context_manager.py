import re


def clean_response(response):
    response = re.sub('<[^>]+>', '', response)
    response = response.strip()
    return response


class DialogContextManager:
    def __init__(self, model, tokenizer, max_input_len, device, style_transfer_model=None):
        self.context = []
        self.context_pretty = []
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.speaker_str = 'Person1'
        self.style_transfer_model = style_transfer_model
        self.device = device

    def write(self, message, debug=False, num_beams=1):
        self.context.append(f"{self.speaker_str}:{message}")
        self.context_pretty.append(f"Question: {message}")

        response = self._generate_answer(self.context, self.model, num_beams)

        if self.style_transfer_model is not None:
            if debug:
                print("generic response: ", response)
            response = self._generate_answer(response.replace("Person2:", ""), self.style_transfer_model)

        self.context.append(response)
        self.context_pretty.append(f"Answer: {response}")
        return "\n".join(self.context_pretty)

    def _generate_answer(self, question, model, num_beams=1):
        inputs_encoding = self.tokenizer(
            "".join(question),
            add_special_tokens=True,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        generate_ids = model.model.generate(
            input_ids=inputs_encoding["input_ids"].to(self.device),
            attention_mask=inputs_encoding["attention_mask"].to(self.device),
            max_length=self.max_input_len,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_beams=num_beams
        )

        preds = [
            self.tokenizer.decode(gen_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            for gen_id in generate_ids
        ]

        response = clean_response("".join(preds))
        return response

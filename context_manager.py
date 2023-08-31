import re


def clean_response(response):
    response = re.sub('<[^>]+>', '', response)
    response = response.strip()
    return response


class DialogContextManager:
    def __init__(self, model, tokenizer, max_input_len, device, style_transfer_model=None,
                 sentence_completion_model=False):
        """
        :param model:
        This class supports only three type of models:
        1. Daily-dialog
        2. Style-transfer
        3. Sentence completion (fine-tuning)
        It will handel a dialog with chosen model while preserving the context of previous questions and answers.
        :param tokenizer:
        :param max_input_len:
        :param device: CUDA or CPU
        :param style_transfer_model: pass generated output to another model which will perform the style transfer
        :param sentence_completion_model: because of how sentence completion model was trained, we will want to add
        "Hi Michael," to the input. otherwise, it's a daily dialog model and from the same reasons we will add "Person1:"
        """
        self.context = []
        self.context_pretty = []
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.speaker_str = "Hi Michael," if sentence_completion_model else 'Person1:'
        self.style_transfer_model = style_transfer_model
        self.device = device
        self.sentence_completion_model = sentence_completion_model

    def write(self, message, debug=False, **kwargs):
        """
        write a question to model
        :param temperature: see _generate_answer()
        :param message:
        :param debug: if using style transfer model, and you wish to see the question before transferring the style.
        :param num_beams: see _generate_answer()
        :return:
        """
        self.context.append(f"{self.speaker_str}{message}")
        self.context_pretty.append(f"Question: {message}")

        response = self._generate_answer(self.context, self.model,**kwargs)

        if self.style_transfer_model is not None:
            if debug:
                print("generic response: ", response)
            response = self._generate_answer(response.replace("Person2:", ""), self.style_transfer_model, temperature)

        self.context.append(response)
        self.context_pretty.append(response if self.sentence_completion_model else f"Answer: {response}")
        return "\n".join(self.context_pretty)

    def _generate_answer(self, question, model,**kwargs):
        """
        This method encodes the given question, and using generate() of Pytorch,
        performs a forward pass, and calculating the final prediction based on the logits.
        finally it performs the decoding.
        :param question:
        :param model:
        :param num_beams: the number of beams to use in beam search in order to calculate probabilities
        :param temperature: value between 0-1. lowering down will relax the difference between different probabilities.
        :return:
        """
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
            **kwargs
        )

        preds = [
            self.tokenizer.decode(gen_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            for gen_id in generate_ids
        ]

        response = clean_response("".join(preds))
        return response

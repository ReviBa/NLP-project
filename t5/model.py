import pytorch_lightning as pl
from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


class T5Model(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=3e-4)

    def load_from_checkpoint_with_custom_tokenizer(self, tokenizer, checkpoint_path):
        self.model.resize_token_embeddings(len(tokenizer))
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])
import pytorch_lightning as pl
from t5.data_set import T5DataSet
import torch


class T5DataLoader(pl.LightningDataModule):
    def __init__(self, df_train, df_test, tokenizer, input_max_len, out_max_len, train_batch_size, test_batch_size):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.out_max_len = out_max_len
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size

    def setup(self):
        self.train_data = T5DataSet(
            self.df_train.input.values,
            self.df_train.target.values,
            self.tokenizer,
            self.input_max_len,
            self.out_max_len
        )

        self.valid_data = T5DataSet(
            self.df_test.input.values,
            self.df_test.target.values,
            self.tokenizer,
            self.input_max_len,
            self.out_max_len
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.test_batch_size
        )

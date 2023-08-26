from t5.consts import TEST_SIZE, RANDOM_STATE
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from t5.data_loader import T5DataLoader
import pytorch_lightning as pl
from t5.model import T5Model


def t5_train(data, device, train_from_checkpoint, save_to_checkpoint, tokenizer, base_model_name, input_max_len, out_max_len, train_batch_size,
             test_batch_size, epochs):
    df_train, df_test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    dataload = T5DataLoader(df_train, df_test, tokenizer, input_max_len, out_max_len, train_batch_size, test_batch_size)
    dataload.setup()
    device = device
    model = T5Model(base_model_name)
    model.load_from_checkpoint_with_custom_tokenizer(tokenizer, train_from_checkpoint)
    model.to(device)

    checkpoint = ModelCheckpoint(  # saving the stats of the model into directory
        dirpath="/content/drive/MyDrive/",
        filename=save_to_checkpoint,
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    trainer = pl.Trainer(
        callbacks=checkpoint,
        max_epochs=epochs,
        accelerator="gpu",
        gpus=1
    )

    trainer.fit(model, dataload)

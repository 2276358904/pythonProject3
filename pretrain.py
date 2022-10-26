import os.path

import tensorflow as tf

from datetime import datetime
from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
from modeling import FasterBERTForMaskedLM
from model_config import ModelConfig


def load_raw_dataset(path=None, name=None):
    if path is None:
        return None
    dataset = load_dataset(path=path, name=name, split="train[:10000]")
    return dataset


def preprocess_function(data):
    block_size = 256
    concatenated_data = {k: sum(v, []) for k, v in data.items()}
    length = len(concatenated_data[list(data.keys())[0]])
    length = length // block_size * block_size
    result = {
        k: [v[i: i + block_size] for i in range(0, length, block_size)] for k, v in concatenated_data.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def convert_to_tf_dataset(dataset, tokenizer:BertTokenizerFast, data_collator):
    tokenized_dataset = dataset.map(
        lambda data: tokenizer(dataset["text"]),
        batched=True,
        remove_columns=["text"])
    preprocessed_dataset = tokenized_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        num_proc=4
    )
    tf_train_dataset = preprocessed_dataset.to_tf_dataset(
        batch_size=64,
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
        shuffle=True,
        collate_fn=data_collator
    )
    tf_val_dataset = preprocessed_dataset.to_tf_dataset(
        batch_size=64,
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
        shuffle=False,
        collate_fn=data_collator
    )
    return tf_train_dataset, tf_val_dataset


class CustomizedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warm_up_steps=4000):
        super(CustomizedSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warm_up_steps = warm_up_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warm_up_steps ** (-1.5))
        arg3 = tf.math.rsqrt(self.d_model)
        return arg3 * tf.math.minimum(arg1, arg2)


def train(train_dataset, val_dataset):
    if not os.path.exists("model"):
        os.mkdir("model")
    model_path = "model" + "/checkpoint-{epoch}-{val_loss:.2f}" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        save_freq="epoch"
    )
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    callbacks = [checkpoint_callback, tensorboard_callback]

    config = ModelConfig()
    model = FasterBERTForMaskedLM(config)
    schedule_lr = CustomizedSchedule(128)
    optimizer = tf.optimizers.Adam(learning_rate=schedule_lr)
    model.compile(optimizer=optimizer)
    model.fit(train_dataset, epochs=10, callbacks=callbacks, validation_data=val_dataset)


if __name__ == "__main__":
    ds = load_raw_dataset(path="wikitext", name="wikitext-103-v1")
    t = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dc = DataCollatorForLanguageModeling(t, return_tensors="tf")
    tds, vds = convert_to_tf_dataset(ds, t, dc)
    train(tds, vds)

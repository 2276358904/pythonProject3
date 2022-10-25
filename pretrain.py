import tensorflow as tf

from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling


def load_raw_dataset(path=None, name=None):
    if path is None:
        return None
    dataset = load_dataset(path=path, name=name)
    return dataset


def preprocess_function(data):
    block_size = 1000
    concatenated_data = {k: sum(v, []) for k, v in data.items()}
    length = len(concatenated_data[list(data.keys())[0]])
    length = length // block_size * block_size
    result = {
        k: [v[i: i + block_size] for i in range(0, length, block_size)] for k, v in concatenated_data.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def convert_to_tf_dataset(dataset, tokenizer, data_collator):
    tokenized_dataset = dataset.map(lambda data: tokenizer(dataset["text"]), remove_columns=["text"])
    preprocessed_dataset = tokenized_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        num_proc=4
    )
    tf_dataset = preprocessed_dataset.to_tf_dataset(
        batch_size=64,
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
        shuffle=True,
        collate_fn=data_collator
    )
    return tf_dataset


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


def train(model):
    learning_rate = CustomizedSchedule(128)  # 模型越大，lr调整越小，不宜采用过高的lr
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer)


if __name__ == "__main__":
    ds = load_raw_dataset(name="wikitext", path="wikitext-103-v1")
    t = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dc = DataCollatorForLanguageModeling(t, return_tensors="tf")
    ds = convert_to_tf_dataset(ds, t, dc)
    for ds in ds.take(10):
        print(ds)

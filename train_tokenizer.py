import os

from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    processors,
    decoders
)


def load_data():
    dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="train[:10%]")
    return dataset


def batch_iterator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]


def train(dataset, batch_size, vocab_size):
    tokenizer = Tokenizer(model=models.WordPiece(unk_token="[UNK]"))

    # Set the normalizer of tokenizer.
    tokenizer.normalizer = normalizers.BertNormalizer()

    # Set the pre_tokenizers of tokenizer.
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # Set the trainers of tokenizer, there we should use the same trainer of above tokenizer.
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    tokenizer.train_from_iterator(batch_iterator(dataset, batch_size), trainer=trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    # Set the post processor of tokenizer.
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    return tokenizer


def save(tokenizer, directory, file):
    cwd = os.getcwd()
    if directory not in os.listdir(cwd):
        os.mkdir(directory)
    tokenizer.model.save(folder=directory)
    path = directory + "/" + file
    if file not in os.listdir(directory):
        open(path, "w")
    tokenizer.save(path=path)


if __name__ == "__main__":
    ds = load_data()
    bs = 1000
    vs = 30522
    t = train(ds, bs, vs)
    save(t, "model", "tokenizer.json")

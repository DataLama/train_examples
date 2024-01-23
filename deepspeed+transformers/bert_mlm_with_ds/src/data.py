from typing import List, Tuple, Dict, Union, Callable, Iterable, TypeVar
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def collate_function(batch: List[Tuple[List[int], List[int]]],
                     pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collect a list of masked token indices, and labels, and
    batch them, padding to max length in the batch.
    """
    max_length = max(len(token_ids) for token_ids, _ in batch)
    padded_token_ids = [
        token_ids +
        [pad_token_id for _ in range(0, max_length - len(token_ids))]
        for token_ids, _ in batch
    ]
    padded_labels = [
        labels + [pad_token_id for _ in range(0, max_length - len(labels))]
        for _, labels in batch
    ]
    src_tokens = torch.LongTensor(padded_token_ids)
    tgt_tokens = torch.LongTensor(padded_labels)
    attention_mask = src_tokens.ne(pad_token_id).type_as(src_tokens)
    return {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "attention_mask": attention_mask,
    }


def masking_function(
        text: str,
        tokenizer: TokenizerType,
        mask_prob: float,
        random_replace_prob: float,
        unmask_replace_prob: float,
        max_length: int,
) -> Tuple[List[int], List[int]]:
    """Given a text string, randomly mask wordpieces for Bert MLM
    training.

    Args:
        text (str):
            The input text
        tokenizer (TokenizerType):
            The tokenizer for tokenization
        mask_prob (float):
            What fraction of tokens to mask
        random_replace_prob (float):
            Of the masked tokens, how many should be replaced with
            random tokens (improves performance)
        unmask_replace_prob (float):
            Of the masked tokens, how many should be replaced with
            the original token (improves performance)
        max_length (int):
            The maximum sequence length to consider. Note that for
            Bert style models, this is a function of the number of
            positional embeddings you learn

    Returns:
        Tuple[List[int], List[int]]:
            The masked token ids (based on the tokenizer passed),
            and the output labels (padded with `tokenizer.pad_token_id`)
    """
    # Note: By default, encode does add the BOS and EOS token
    # Disabling that behaviour to make this more clear
    tokenized_ids = ([tokenizer.bos_token_id] +
                     tokenizer.encode(text,
                                      add_special_tokens=False,
                                      truncation=True,
                                      max_length=max_length - 2) +
                     [tokenizer.eos_token_id])
    seq_len = len(tokenized_ids)
    tokenized_ids = np.array(tokenized_ids)
    subword_mask = np.full(len(tokenized_ids), False)

    # Masking the BOS and EOS token leads to slightly worse performance
    low = 1
    high = len(subword_mask) - 1
    mask_choices = np.arange(low, high)
    num_subwords_to_mask = max(
        int((mask_prob * (high - low)) + np.random.rand()), 1)
    subword_mask[np.random.choice(mask_choices,
                                  num_subwords_to_mask,
                                  replace=False)] = True

    # Create the labels first
    labels = np.full(seq_len, tokenizer.pad_token_id)
    labels[subword_mask] = tokenized_ids[subword_mask]

    tokenized_ids[subword_mask] = tokenizer.mask_token_id

    # Now of the masked tokens, choose how many to replace with random and how many to unmask
    rand_or_unmask_prob = random_replace_prob + unmask_replace_prob
    if rand_or_unmask_prob > 0:
        rand_or_unmask = subword_mask & (np.random.rand(len(tokenized_ids)) <
                                         rand_or_unmask_prob)
        if random_replace_prob == 0:
            unmask = rand_or_unmask
            rand_mask = None
        elif unmask_replace_prob == 0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = unmask_replace_prob / rand_or_unmask_prob
            decision = np.random.rand(len(tokenized_ids)) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
        if unmask is not None:
            tokenized_ids[unmask] = labels[unmask]
        if rand_mask is not None:
            weights = np.ones(tokenizer.vocab_size)
            weights[tokenizer.all_special_ids] = 0
            probs = weights / weights.sum()
            num_rand = rand_mask.sum()
            tokenized_ids[rand_mask] = np.random.choice(tokenizer.vocab_size,
                                                        num_rand,
                                                        p=probs)
    return tokenized_ids.tolist(), labels.tolist()


class WikiTextMLMDataset(Dataset):
    """A [Map style dataset](https://pytorch.org/docs/stable/data.html)
    for iterating over the wikitext dataset. Note that this assumes
    the dataset can fit in memory. For larger datasets
    you'd want to shard them and use an iterable dataset (eg: see
    [Infinibatch](https://github.com/microsoft/infinibatch))

    Args:
        Dataset (datasets.arrow_dataset.Dataset):
            The wikitext dataset
        masking_function (Callable[[str], Tuple[List[int], List[int]]])
            The masking function. To generate one training instance,
            the masking function is applied to the `text` of a dataset
            record

    """
    def __init__(
        self,
        dataset: datasets.arrow_dataset.Dataset,
        masking_function: Callable[[str], Tuple[List[int], List[int]]],
    ) -> None:
        self.dataset = dataset
        self.masking_function = masking_function

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        tokens, labels = self.masking_function(self.dataset[idx]["text"])
        return (tokens, labels)


T = TypeVar("T")


class InfiniteIterator(object):
    def __init__(self, iterable: Iterable[T]) -> None:
        self._iterable = iterable
        self._iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self) -> T:
        next_item = None
        try:
            next_item = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._iterable)
            next_item = next(self._iterator)
        return next_item


def create_data_iterator(
        mask_prob: float,
        random_replace_prob: float,
        unmask_replace_prob: float,
        batch_size: int,
        max_seq_length: int = 512,
        tokenizer: str = "roberta-base",
) -> InfiniteIterator:
    """Create the dataloader.

    Args:
        mask_prob (float):
            Fraction of tokens to mask
        random_replace_prob (float):
            Fraction of masked tokens to replace with random token
        unmask_replace_prob (float):
            Fraction of masked tokens to replace with the actual token
        batch_size (int):
            The batch size of the generated tensors
        max_seq_length (int, optional):
            The maximum sequence length for the MLM task. Defaults to 512.
        tokenizer (str, optional):
            The tokenizer to use. Defaults to "roberta-base".

    Returns:
        InfiniteIterator:
            The torch DataLoader, wrapped in an InfiniteIterator class, to
            be able to continuously generate samples

    """
    wikitext_dataset = datasets.load_dataset("wikitext",
                                             "wikitext-2-v1",
                                             split="train")
    wikitext_dataset = wikitext_dataset.filter(
        lambda record: record["text"] != "").map(
            lambda record: {"text": record["text"].rstrip("\n")})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    masking_function_partial = partial(
        masking_function,
        tokenizer=tokenizer,
        mask_prob=mask_prob,
        random_replace_prob=random_replace_prob,
        unmask_replace_prob=unmask_replace_prob,
        max_length=max_seq_length,
    )
    dataset = WikiTextMLMDataset(wikitext_dataset, masking_function_partial)
    collate_fn_partial = partial(collate_function,
                                 pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn_partial)

    return InfiniteIterator(dataloader)

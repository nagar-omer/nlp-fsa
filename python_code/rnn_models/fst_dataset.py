from torch import Tensor
from torch.utils.data import Dataset
from random import shuffle
import numpy as np
from fst_tools import FSTools
from rnn_models.dataset_params import FSTParams

PAD = "_PAD_"


class FstDataset(Dataset):
    def __init__(self, parmas: FSTParams):
        self._fst = FSTools().rand_fst(parmas.FST_STATES_SIZE, parmas.FST_ALPHABET_SIZE, parmas.FST_ACCEPT_STATES_SIZE)
        self._chr_embed = self._get_embeddings(self._fst.alphabet)                   # index alphabet for embeddings
        self._data = self._build_data(parmas.DATASET_SIZE, parmas.NEGATIVE_SAMPLES)  # get data

    def _get_embeddings(self, alphabet):
        embed = {symbol: i for i, symbol in enumerate(alphabet)}
        embed[PAD] = len(embed)                                     # special index for padding
        return embed

    def _build_data(self, size, negative):
        positive_size = size // 2 if negative else 0
        negative_size = size - positive_size
        positive = []
        negative = []
        # add positive samples
        for _ in range(positive_size * 2):
            # data.append(([self._chr_embed[symbol] for symbol in self._fst.go()], 1))
            positive.append([self._chr_embed[symbol] for symbol in self._fst.go()])
        # add negative samples
        positive = [(list(x), 1) for x in set(tuple(x) for x in positive)][:positive_size]
        for i in range(negative_size * 2):
            # data.append(([self._chr_embed[symbol] for symbol in
            #               self._fst.generate_negative(max_size=len(data[i][0]) + 1)], 0))
            negative.append([self._chr_embed[symbol] for symbol in
                             self._fst.generate_negative(sample_len=len(positive[i % len(positive)][0]) + 1)])
        negative = [(list(x), 0) for x in set(tuple(x) for x in negative)][:negative_size]
        data = negative + positive
        shuffle(data)
        return data

    # function for torch Dataloader - creates batch matrices using Padding
    def collate_fn(self, batch):
        lengths_sequences = []
        # calculate max word len + max char len
        for sample, label in batch:
            lengths_sequences.append(len(sample))

        # in order to pad all batch to a single dimension max length is needed
        max_lengths_sequences = np.max(lengths_sequences)

        # new batch variables
        lengths_sequences_batch = []
        labels_batch = []
        for sample, label in batch:
            # pad word vectors
            lengths_sequences_batch.append([self._chr_embed[PAD]] * (max_lengths_sequences - len(sample)) + sample)
            labels_batch.append(label)

        return Tensor(lengths_sequences_batch).long(), Tensor(labels_batch).long()

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = FstDataset(FSTParams())
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )

    for i, (sequence, label) in enumerate(dl):
        print(i, sequence, label)
    e = 0


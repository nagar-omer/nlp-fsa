from torch.nn import Module, LSTM, Dropout, Embedding, AvgPool1d, Linear, MaxPool1d, MaxPool2d
from torch import sigmoid
from binary_params import SequenceEncoderParams, MLPParams, BinaryModuleParams


class SequenceEncoderModule(Module):
    def __init__(self, params: SequenceEncoderParams):
        super(SequenceEncoderModule, self).__init__()
        # word embed layer
        self._embeddings = Embedding(params.EMBED_vocab_dim, params.EMBED_dim)
        # Bi-LSTM layers
        self._lstm = LSTM(params.EMBED_dim, params.LSTM_hidden_dim // 2, params.LSTM_layers, batch_first=True,
                          bidirectional=True, dropout=params.LSTM_dropout)

    def forward(self, words_embed):
        # max pool defined by len of sequence
        max_pool = MaxPool1d(words_embed.shape[1], 1)
        # embed_word -> LSTM layer -> MAX-POOL
        x = self._embeddings(words_embed)
        # 3 layers LSTM
        output_seq, _ = self._lstm(x)
        # max pool
        return max_pool(output_seq.transpose(1, 2)).squeeze(dim=2)
        # return output_seq[:, -1, :]


class MLPModule(Module):
    def __init__(self, params: MLPParams):
        super(MLPModule, self).__init__()
        # useful info in forward function
        self._linear = Linear(params.LINEAR_in_dim, params.LINEAR_out_dim)

    def forward(self, x):
        x = self._linear(x)
        x = sigmoid(x)
        return x


class BinaryModule(Module):
    def __init__(self, params: BinaryModuleParams):
        super(BinaryModule, self).__init__()
        # useful info in forward function
        self._sequence_lstm = SequenceEncoderModule(params.SEQUENCE_PARAMS)
        self._mlp = MLPModule(params.LINEAR_PARAMS)
        self.optimizer = self.set_optimizer(params.LEARNING_RATE, params.OPTIMIZER, params.REGULARIZATION)

    def set_optimizer(self, lr, opt, l2_reg):
        return opt(self.parameters(), lr=lr, weight_decay=l2_reg)

    def forward(self, x):
        x = self._sequence_lstm(x)
        x = self._mlp(x)
        return x


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset_params import FSTParams
    from fst_dataset import FstDataset

    _fst_params = FSTParams()
    _ds = FstDataset(_fst_params)
    _dl = DataLoader(
        dataset=_ds,
        batch_size=64,
        collate_fn=_ds.collate_fn
    )
    _binary_module = BinaryModule(BinaryModuleParams(alphabet_size=len(_ds.chr_embed),
                                                     lstm_out_dim=100))
    for _i, (_sequence, _label) in enumerate(_dl):
        _out = _binary_module(_sequence)
        print(_out)
        e = 0


from torch.nn import Module, LSTM, Dropout, Embedding, AvgPool1d, Linear, MaxPool1d
from torch import sigmoid

from binary_params import SequenceEncoderParams, MLPParams, BinaryModuleParams


class SequenceEncoderModule(Module):
    def __init__(self, params: SequenceEncoderParams):
        super(SequenceEncoderModule, self).__init__()
        # word embed layer
        self._embeddings = Embedding(params.EMBED_vocab_dim, params.EMBED_dim)
        # Bi-LSTM layers
        self._lstm = LSTM(params.EMBED_dim, params.LSTM_hidden_dim, params.LSTM_layers, batch_first=True,
                          bidirectional=False)
        self._dropout = Dropout(p=params.LSTM_dropout)

    def forward(self, words_embed):
        # embed_word -> LSTM layer -> AVG-POOL
        activate_max_pool = MaxPool1d(words_embed.shape[1], 1)
        x = self._embeddings(words_embed)
        # 3 layers LSTM
        output_seq, _ = self._lstm(self._dropout(x))
        # avg pool
        # return activate_max_pool(output_seq.transpose(1, 2)).squeeze(dim=2)
        return output_seq[:, -1, :]


class MLPModule(Module):
    def __init__(self, params: MLPParams):
        super(MLPModule, self).__init__()
        # useful info in forward function
        self._layer0 = Linear(params.LINEAR_in_dim, params.LINEAR_hidden_dim_0)
        self._layer1 = Linear(params.LINEAR_hidden_dim_0, params.LINEAR_out_dim)
        self._activation = params.Activation

    def forward(self, x):
        x = self._layer0(x)
        x = self._activation(x)
        x = self._layer1(x)
        x = sigmoid(x)
        return x


class BinaryModule(Module):
    def __init__(self, params: BinaryModuleParams):
        super(BinaryModule, self).__init__()
        # useful info in forward function
        self._sequence_lstm = SequenceEncoderModule(SequenceEncoderParams())
        self._mlp = MLPModule(MLPParams())
        self.optimizer = self.set_optimizer(params.LEARNING_RATE, params.OPTIMIZER)

    def set_optimizer(self, lr, opt):
        return opt(self.parameters(), lr=lr)

    def forward(self, x):
        x = self._sequence_lstm(x)
        x = self._mlp(x)
        return x


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset_params import FSTParams
    from fst_dataset import FstDataset

    _ds = FstDataset(FSTParams())
    _dl = DataLoader(
        dataset=_ds,
        batch_size=64,
        collate_fn=_ds.collate_fn
    )
    _binary_module = BinaryModule(BinaryModuleParams())
    for _i, (_sequence, _label) in enumerate(_dl):
        _out = _binary_module(_sequence)
        print(_out)
        e = 0


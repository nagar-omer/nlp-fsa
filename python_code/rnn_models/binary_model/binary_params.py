from torch.nn.functional import relu, binary_cross_entropy
from torch.optim import Adam
from dataset_params import FSTParams


ALPHABET_SIZE = 3
class BinaryFSTParams(FSTParams):
    def __init__(self):
        super().__init__()
        self.DATASET_SIZE = 10000
        self.NEGATIVE_SAMPLES = True
        self.FST_ALPHABET_SIZE = ALPHABET_SIZE
        self.FST_STATES_SIZE = 5
        self.FST_ACCEPT_STATES_SIZE = 1


LSTM_OUT_DIM = 100
class SequenceEncoderParams:
    def __init__(self):
        self.EMBED_dim = 10
        self.EMBED_vocab_dim = ALPHABET_SIZE + 1    # +1 for _PAD_
        self.LSTM_hidden_dim = LSTM_OUT_DIM
        self.LSTM_layers = 3
        self.LSTM_dropout = 0.1


class MLPParams:
    def __init__(self):
        self.LINEAR_in_dim = LSTM_OUT_DIM
        self.LINEAR_hidden_dim_0 = 50
        self.LINEAR_out_dim = 1
        self.Activation = relu


class BinaryModuleParams:
    def __init__(self):
        self.SEQUENCE_PARAMS = SequenceEncoderParams()
        self.LINEAR_PARAMS = MLPParams()
        self.LEARNING_RATE = 1e-3
        self.OPTIMIZER = Adam


class BinaryActivatorParams:
    def __init__(self):
        self.TRAIN_TEST_SPLIT = 0.8
        self.LOSS = binary_cross_entropy
        self.BATCH_SIZE = 64
        self.GPU = True
        self.EPOCHS = 10
        self.VALIDATION_RATE = 200

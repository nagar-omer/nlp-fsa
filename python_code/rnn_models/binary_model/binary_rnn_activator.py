from sys import stdout
from bokeh.plotting import figure, show
from torch.utils.data import DataLoader, random_split

from binary_params import BinaryActivatorParams
from binary_rnn_model import BinaryModule
from fst_dataset import FstDataset

TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
ACCURACY_PLOT = "accuracy"


class binaryActivator:
    def __init__(self, model: BinaryModule, params: BinaryActivatorParams, data: FstDataset):
        self._model = model.cuda() if params.GPU else model
        self._gpu = params.GPU
        self._epochs = params.EPOCHS
        self._batch_size = params.BATCH_SIZE
        self._loss_func = params.LOSS
        self._load_data(data, params.TRAIN_TEST_SPLIT, params.BATCH_SIZE)
        self._init_loss_and_acc_vec()
        self._init_print_att()

    # init loss and accuracy vectors (as function of epochs)
    def _init_loss_and_acc_vec(self):
        self._loss_vec_train = []
        self._loss_vec_dev = []
        self._accuracy_vec_train = []
        self._accuracy_vec_dev = []

    # init variables that holds the last update for loss and accuracy
    def _init_print_att(self):
        self._print_train_accuracy = 0
        self._print_train_loss = 0
        self._print_dev_accuracy = 0
        self._print_dev_loss = 0

    # update loss after validating
    def _update_loss(self, loss, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._loss_vec_train.append(loss)
            self._print_train_loss = loss
        elif job == DEV_JOB:
            self._loss_vec_dev.append(loss)
            self._print_dev_loss = loss

    # update accuracy after validating
    def _update_accuracy(self, pred, true, job=TRAIN_JOB):
        # calculate acc
        acc = sum([1 if int(i) == int(j) else 0 for i, j in zip(pred, true)]) / len(pred)
        if job == TRAIN_JOB:
            self._print_train_accuracy = acc
            self._accuracy_vec_train.append(acc)
            return acc
        elif job == DEV_JOB:
            self._print_dev_accuracy = acc
            self._accuracy_vec_dev.append(acc)
            return acc

    # print progress of a single epoch as a percentage
    def _print_progress(self, batch_index, len_data, job=""):
        prog = int(100 * (batch_index + 1) / len_data)
        stdout.write("\r\r\r\r\r\r\r\r" + job + " %d" % prog + "%")
        print("", end="\n" if prog == 100 else "")
        stdout.flush()

    # print last loss and accuracy
    def _print_info(self, jobs=()):
        if TRAIN_JOB in jobs:
            print("Acc_Train: " + '{:{width}.{prec}f}'.format(self._print_train_accuracy, width=6, prec=4) +
                  " || Loss_Train: " + '{:{width}.{prec}f}'.format(self._print_train_loss, width=6, prec=4),
                  end=" || ")
        if DEV_JOB in jobs:
            print("Acc_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_accuracy, width=6, prec=4) +
                  " || Loss_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_loss, width=6, prec=4),
                  end=" || ")
        print("")

    # plot loss / accuracy graph
    def plot_line(self, job=LOSS_PLOT):
        p = figure(plot_width=600, plot_height=250, title="Rand_FST - Dataset " + job,
                   x_axis_label="epochs", y_axis_label=job)
        color1, color2 = ("orange", "red") if job == LOSS_PLOT else ("green", "blue")
        y_axis_train = self._loss_vec_train if job == LOSS_PLOT else self._accuracy_vec_train
        y_axis_dev = self._loss_vec_dev if job == LOSS_PLOT else self._accuracy_vec_dev
        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend="dev")
        show(p)

    def _plot_acc_dev(self):
        self.plot_line(LOSS_PLOT)
        self.plot_line(ACCURACY_PLOT)

    @property
    def model(self):
        return self._model

    @property
    def loss_train_vec(self):
        return self._loss_vec_train

    @property
    def accuracy_train_vec(self):
        return self._accuracy_vec_train

    @property
    def loss_dev_vec(self):
        return self._loss_vec_dev

    @property
    def accuracy_dev_vec(self):
        return self._accuracy_vec_dev

    # load dataset
    def _load_data(self, train_dataset, train_split, batch_size):
        # calculate lengths off train and dev according to split ~ (0,1)
        len_train = int(len(train_dataset) * train_split)
        len_dev = len(train_dataset) - len_train
        # split dataset
        train, dev = random_split(train_dataset, (len_train, len_dev))
        # set train loader
        self._train_loader = DataLoader(
            train,
            batch_size=64,
            collate_fn=train_dataset.collate_fn,
            shuffle=True
        )
        # set validation loader
        self._dev_loader = DataLoader(
            dev,
            batch_size=64,
            collate_fn=train_dataset.collate_fn,
            shuffle=True
        )

    def _to_gpu(self, x, l):
        x = x.cuda() if self._gpu else x
        l = l.cuda() if self._gpu else l
        return x, l

    # train a model, input is the enum of the model type
    def train(self, show_plot=True):
        self._init_loss_and_acc_vec()
        # calc number of iteration in current epoch
        len_data = len(self._train_loader)
        for epoch_num in range(self._epochs):
            # calc number of iteration in current epoch
            for batch_index, (sequence, label) in enumerate(self._train_loader):
                sequence, label = self._to_gpu(sequence, label)
                # print progress
                self._model.train()

                output = self._model(sequence)                  # calc output of current model on the current batch
                loss = self._loss_func(output.squeeze(dim=0), label.unsqueeze(dim=1).float())  # calculate loss
                loss.backward()                                 # back propagation

                if (batch_index + 1) % self._batch_size == 0 or (batch_index + 1) == len_data:  # batching
                    self._model.optimizer.step()                # update weights
                    self._model.zero_grad()                     # zero gradients
                self._print_progress(batch_index, len_data, job=TRAIN_JOB)
            # validate and print progress
            self._validate(self._train_loader, job=TRAIN_JOB)
            self._validate(self._dev_loader, job=DEV_JOB)
            self._print_info(jobs=[TRAIN_JOB, DEV_JOB])

        if show_plot:
            self._plot_acc_dev()

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        # for calculating total loss and accuracy
        loss_count = 0
        true_labels = []
        pred_labels = []

        self._model.eval()
        # calc number of iteration in current epoch
        len_data = len(data_loader)
        for batch_index, (sequence, label) in enumerate(data_loader):
            sequence, label = self._to_gpu(sequence, label)
            # print progress
            self._print_progress(batch_index, len_data, job=VALIDATE_JOB)
            output = self._model(sequence)
            # calculate total loss
            loss_count += self._loss_func(output.squeeze(dim=0), label.unsqueeze(dim=1).float())
            true_labels += label.tolist()
            pred_labels += output.squeeze().round().long().tolist()

        # update loss accuracy
        loss = float(loss_count / len(data_loader))
        self._update_loss(loss, job=job)
        self._update_accuracy(pred_labels, true_labels, job=job)
        return loss


if __name__ == '__main__':
    from binary_params import BinaryFSTParams, BinaryModuleParams
    activator = binaryActivator(BinaryModule(BinaryModuleParams()), BinaryActivatorParams(),
                                FstDataset(BinaryFSTParams()))
    activator.train()

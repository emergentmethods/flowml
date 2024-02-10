import torch
from flowdapt.lib.logger import get_logger
import math

logger = get_logger(__name__)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the
        # positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class WindowDataset(torch.utils.data.Dataset):
    """
    Given the lookback (window_size), traverses the data
    in reverse order and returns a the X and y on __getitem__
    """
    def __init__(self, xs, ys, window_size):
        self.xs = xs
        self.ys = ys
        self.window_size = window_size

    def __len__(self):
        return len(self.xs) - self.window_size

    def __getitem__(self, index):
        idx_rev = len(self.xs) - self.window_size - index - 1
        window_x = torch.flatten(self.xs[idx_rev:idx_rev+self.window_size, :])
        # Beware of indexing, these two window_x and window_y are aimed at the same row!
        # this is what happens when you use ":"
        window_y = self.ys[idx_rev+self.window_size-1, :]
        return window_x, window_y


class WindowDatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, xs, ys, window_size):
        self.xs = xs
        self.ys = ys
        self.window_size = window_size

    def __len__(self):
        return len(self.xs) - self.window_size

    def __getitem__(self, index):
        idx_rev = len(self.xs) - self.window_size - index - 1
        window_x = self.xs[idx_rev:idx_rev+self.window_size, :]
        # Beware of indexing, these two window_x and window_y are aimed at the same row!
        # this is what happens when you use :
        window_y = self.ys[idx_rev+self.window_size-1, :].unsqueeze(0)
        return window_x, window_y


class WindowDatasetTransformerTraditional(torch.utils.data.Dataset):
    def __init__(self, xs, ys, window_size, rollout=6):
        self.xs = xs
        self.ys = ys
        self.window_size = window_size
        self.rollout = rollout

    def __len__(self):
        return len(self.xs) - self.window_size - self.rollout - 1

    def __getitem__(self, index):
        idx_rev = len(self.xs) - self.window_size - index - 1
        window_x = self.xs[idx_rev:idx_rev+self.window_size, :]
        # Beware of indexing, these two window_x and window_y are aimed at the same row!
        # this is what happens when you use :
        window_y = self.ys[idx_rev+self.window_size:self.rollout, :]
        return window_x, window_y


class WindowAutoRegressiveDataset(torch.utils.data.Dataset):
    def __init__(self, xs, window_size, rollout, valid: bool = False):
        self.xs = xs
        self.window_size = window_size
        self.rollout = rollout
        self.validation = valid

    def __len__(self):
        if not self.validation:
            return len(self.xs) - self.window_size - self.rollout - 1
        # number of samples for training is the total points divided by the
        # full rollout for a single rollout
        else:
            return len(self.xs) - self.window_size - self.rollout - 1
            # return int(math.floor(len(self.xs) / (self.window_size + self.rollout)))
        # else:
        #     # number of samples is as many as possible for validation
        #     return len(self.xs) - self.window_size - self.rollout - 1

    def __getitem__(self, index):

        if self.validation:
            idx = index  # * (self.window_size + self.rollout)
        else:
            idx = index

        # idx_rev = len(self.xs) - self.window_size - self.rollout - index - 1
        window_x = self.xs[idx:idx+self.window_size, :]
        # Beware of indexing, these two window_x and window_y are aimed at subsequent rows!
        # this is what happens when you use : for indexing
        y_start = idx + self.window_size
        y_end = y_start + self.rollout
        window_y = self.xs[y_start:y_end, :]
        return window_x, window_y


class EarlyStopping:
    """Stops early if the validation loss has plateaued"""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.debug(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def check_for_gpu():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        logger.debug("Found cuda enabled gpu")
    else:
        device = torch.device("cpu")

    return device


def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)

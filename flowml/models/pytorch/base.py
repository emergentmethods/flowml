import torch
from typing import Any, Callable
from abc import abstractmethod
from io import BytesIO

from flowml.models.base import BaseMLModel
from flowdapt.compute.artifacts import Artifact
from flowml.models.pytorch.utils import check_for_gpu, set_num_threads
from flowdapt.lib.logger import get_logger
# from flowml.tensorboard import TensorboardLogger
import pandas as pd
import numpy.typing as npt
from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
from flowml.models.pytorch.architectures.mlp import Net
import numpy as np
import tempfile
import os

logger = get_logger(__name__)


class BasePyTorchModel(BaseMLModel):
    def __init_model__(
            self,
            epochs: int = 10,
            batch_size: int = 32,
            lookback: int = 10,
            hidden_dim: int = 32,
            num_layers: int = 2,
            dropout: float = 0.1,
            nhead: int = 8,
            shuffle: bool = False,
            extra_train_params: dict[str, Any] = {},
            logger: Callable | None = None,
            early_stopping: bool = False,
            num_threads: int = 1,
            **kwargs
    ) -> "BasePyTorchModel":
        self.num_threads = num_threads
        self._device = check_for_gpu()
        set_num_threads(num_threads)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lookback = lookback
        self.shuffle = shuffle
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.nhead = nhead
        self._reset_logger = True
        self.extra_train_params = extra_train_params
        if self.default_logger:
            self.logger = SummaryWriter
        else:
            self.logger = logger
        self.tb_logger = None
        self.optimizer: torch.optim.Optimizer | None = None  # torch.optim.Adam()
        self.nn: torch.nn.Module = Net()
        self.best_loss: float | None = None
        self.hn = None
        self.cn = None
        self.early_stopping = early_stopping

        return self

    def get_params(self):
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lookback": self.lookback,
            "hidden_dim": self.hidden_dim,
            "shuffle": self.shuffle,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "nhead": self.nhead,
            "extra_train_params": self.extra_train_params,
            "default_logger": False,
            "num_threads": self.num_threads
        }

    # This is required for the model to be added to ClusterMemory as-is,
    # while keeping it portable to be trained on a GPU but used on a CPU
    # if needed. ClusterMemory *ALWAYS* uses some type of pickling for over
    # the wire communication.
    def __getstate__(self):
        buffer = BytesIO()
        self.logger = None
        # Serialize the state
        state = {
            "params": self.get_params(),
            "name": self.name_str,
            "best_loss": self.best_loss,
            "namespace": self._namespace,
            "network_type": self.nn,
            "optimizer_type": self.optimizer,
            # "network_statedict": self.nn.state_dict() if self.nn else None,
            # "optimizer_statedict": self.optimizer.state_dict() if self.optimizer else None,
        }

        # Use torch save so we know it's compatible with the model
        torch.save(state, buffer)
        return buffer.getvalue()

    def __setstate__(self, bytear):
        # Load the state and make sure it's loaded on CPU
        buffer = BytesIO(bytear)

        _device = check_for_gpu()

        if _device == torch.device("cpu"):
            state = torch.load(buffer, map_location=_device)
        else:
            state = torch.load(buffer)

        set_num_threads(state["params"]["num_threads"])
        # Reinitialize the model
        self.__init__(state["name"], state["namespace"], **state["params"])

        self.nn = state["network_type"]
        self.optimizer = state["optimizer_type"]

        if _device == torch.device("cuda"):
            self.nn.to(_device)

        # network_statedict = state["network_statedict"]
        # optimizer_statedict = state["optimizer_statedict"]

        # self.nn.load_state_dict(network_statedict)
        # self.optimizer.load_state_dict(optimizer_statedict)

        self.best_loss = state["best_loss"]

    @classmethod
    def from_artifact(cls, **kwargs) -> Callable[[Artifact], Any]:
        def _inner(artifact: Artifact):
            # Require model artifact
            if not artifact.get_meta("type", None) == "model":
                raise ValueError("Artifact must be of type `model`")

            # Get the meta from the Artifact
            model_params = artifact.get_meta("params", {})
            study_id = artifact.get_meta("namespace", None)

            # Update the model params with the ones from the artifact
            kwargs.update(model_params)
            file = artifact.get_file("model.pt")

            _device = check_for_gpu()
            set_num_threads(model_params["num_threads"])

            # Create the instance and assign the model
            instance = cls(name=artifact.name, **model_params)

            # Load the model and optimizer
            with file.open(mode="rb") as f:
                if _device == torch.device("cpu"):
                    checkpoint = torch.load(f, map_location=_device)
                else:
                    checkpoint = torch.load(f)

                instance.nn = checkpoint["network_architecture"]
                instance.optimizer = checkpoint["optimizer"]

                # instance.nn.load_state_dict(checkpoint["model_state_dict"])
                # instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                instance.best_loss = checkpoint["best_loss"]

            if _device == torch.device("cuda"):
                instance.nn.to(_device)
            instance._device = _device
            instance._namespace = study_id

            return instance
        return _inner

    def to_artifact(self, **kwargs) -> Callable[[Artifact, Any], None]:
        def _inner(artifact: Artifact, value: Any):
            assert self.nn and self.optimizer, "Model must be trained before saving"

            # Add the meta to the artifact
            artifact["type"] = "model"
            artifact["params"] = self.get_params()
            artifact["model_type"] = self.get_type()
            artifact["namespace"] = self._namespace

            with artifact.new_file("model.pt", exist_ok=True).open("wb") as f:
                torch.save(
                    {
                        "network_architecture": self.nn,
                        "best_loss": self.best_loss,
                        "optimizer": self.optimizer,
                        # "model_state_dict": self.nn.state_dict(),
                        # "optimizer_state_dict": self.optimizer.state_dict()
                    },
                    f
                )
        return _inner

    @abstractmethod
    def fit(self, *args, **kwargs):
        ...

    @abstractmethod
    def predict(self, *args, **kwargs):
        ...

    def log(self, name: str, value: Any, step: int):
        """
        Wrapper for the logger callaback
        """
        if self.logger is None:
            return

        if self._reset_logger:
            self.current_log_dir = os.path.join(tempfile.gettempdir(), self.name_str)
            os.makedirs(self.current_log_dir, exist_ok=True)
            self.tb_logger = self.logger(self.current_log_dir)
            self._reset_logger = False

        self.tb_logger.add_scalar(name, value, step)

    def reset_logger(self):
        if self.logger is None:
            return
        self.tb_logger.flush()
        self.tb_logger.close()
        self.logger_to_artifact()
        self.tb_logger = None
        self._reset_logger = True

    def _validate_args(self,
                       xs: npt.ArrayLike | pd.DataFrame,
                       ys: npt.ArrayLike | pd.DataFrame | None = None,
                       eval_set: Any = None
                       ) -> tuple[torch.Tensor, torch.Tensor | None,
                                  torch.Tensor | None, torch.Tensor | None]:
        """
        Validates the incoming arguments to ensure they are numpy arrays
        """
        if isinstance(xs, pd.DataFrame):
            x = torch.from_numpy(xs.values).float()  # xs.to_numpy()
            logger.info(f"Validating {x.shape[0]} samples and {x.shape[1]} features")
        elif xs is None:
            x = None
        elif type(xs) == npt.ArrayLike or type(xs) == np.ndarray:
            x = torch.from_numpy(xs).float()
        else:
            raise ValueError("xs must be a numpy array or pandas dataframe")

        if isinstance(ys, pd.DataFrame):
            y = torch.from_numpy(ys.values).float()  # .to_numpy()
        elif ys is None:
            y = None
        elif type(ys) == npt.ArrayLike or type(ys) == np.ndarray:
            y = torch.from_numpy(ys).float()
        else:
            raise ValueError("ys must be a numpy array or pandas dataframe")

        if eval_set is not None and isinstance(eval_set[0][0], pd.DataFrame):
            valid_xs = torch.from_numpy(eval_set[0][0].values).float()  # to_numpy()
            valid_ys = torch.from_numpy(eval_set[0][1].values).float()
        elif eval_set is None:
            valid_xs = None
            valid_ys = None
        elif type(eval_set[0][0]) == npt.ArrayLike or type(eval_set[0][0]) == np.ndarray:
            valid_xs = torch.from_numpy(eval_set[0][0]).float()
            valid_ys = torch.from_numpy(eval_set[0][1]).float()
        else:
            raise ValueError("eval_set must be a numpy array or pandas dataframe")

        return x, y, valid_xs, valid_ys

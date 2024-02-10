import torch
from typing import Any
import numpy.typing as npt
from flowml.models.pytorch.base import BasePyTorchModel
import pandas as pd
from flowdapt.lib.logger import get_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from flowml.models.pytorch.utils import EarlyStopping, WindowDataset
from flowml.models.pytorch.architectures.mlp import Net

logger = get_logger(__name__)


class PyTorchMLPRegressor(BasePyTorchModel):

    def fit(self,
            xs: npt.ArrayLike | pd.DataFrame,
            ys: npt.ArrayLike | pd.DataFrame,
            eval_set: Any = None):
        """
        High level wrapper that takes the xs, ys, and eval_set and trains
        a model.
        """
        x, y, valid_x, valid_y = self._validate_args(xs, ys, eval_set)

        if x is None or y is None:
            logger.warning("No training data provided, skipping fit")

        dataset = self.make_dataloader(x, y, self.lookback)
        if valid_x is not None:
            valid_dataset = self.make_dataloader(valid_x, valid_y, self.lookback)
        else:
            valid_dataset = None

        self.nn = Net(x.shape[1]*self.lookback, y.shape[1], 1).to(self._device)
        self.nn, self.optimizer = self.train_loop(dataset, valid_dataset, self.nn)

        return self

    def predict(self, xs: npt.ArrayLike):
        x, _, _, _ = self._validate_args(xs)
        # logger.info(f"Predicting on {x.shape[0]} samples and {x.shape[1]} features")

        # self.nn.to(self._device)
        self.nn.eval()
        data = torch.flatten(x)
        data = data.to(self._device)
        preds = self.nn(data)
        preds = preds.cpu()
        return preds.detach().numpy()

    def make_dataloader(self,
                        xs: torch.Tensor,
                        ys: torch.Tensor,
                        lookback: int = 10,
                        batch_size: int | None = None):
        """
        Given the xs and ys, make the dataset and dataloader for training.
        """

        dataset = WindowDataset(xs,
                                ys,
                                window_size=lookback)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0,
            shuffle=self.shuffle
        )

        return dataloader

    def train_loop(self, dataloader: Any, valid_dataloader: Any, model: torch.nn.Module):
        """
        Provided the dataloader and valid_dataloader, train the model.
        """
        model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=5e-5, weight_decay=1e-4)
        learning_rate_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        early_stopping = EarlyStopping(patience=50)

        logger.info("Starting training loop")
        min_valid_loss = 1e10
        total_batches = 0
        for epoch in range(0, self.epochs):
            train_loss = 0.
            for batch, batch_data in enumerate(dataloader):
                x, y_target = batch_data
                x = x.float().to(self._device)  # type: ignore
                y_target = y_target.float().to(self._device)
                y_pred = model(x)
                loss = criterion(y_pred, y_target)
                # Backprogation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_batches += 1
                train_loss += loss.item()

            train_loss /= float(batch+1)
            # get loss on valid_dataset
            if valid_dataloader is not None:
                valid_loss = 0.
                model.eval()
                with torch.no_grad():
                    for batch, batch_data in enumerate(valid_dataloader):
                        x, y_target = batch_data
                        x = x.float().to(self._device)
                        y_target = y_target.float().to(self._device)
                        y_pred = model(x)
                        loss = criterion(y_pred, y_target)
                        valid_loss += loss.item()

                valid_loss /= float(batch+1)
                model.train()
                learning_rate_scheduler.step(valid_loss)

                # save best model for later return
                if valid_loss < min_valid_loss:
                    logger.debug(f"found best model with loss {valid_loss}")
                    best_model = model
                    best_optim = optimizer
                    min_valid_loss = valid_loss
                    self.best_loss = float(min_valid_loss)

                if early_stopping(valid_loss) and self.early_stopping:
                    logger.debug("Model is not improving, Early stopping activated")
                    break

                self.log("valid_loss", valid_loss, step=epoch)

            # log epoch based metrics of interest
            self.log("train_loss", train_loss, epoch)

            lrs = []
            for grp in optimizer.param_groups:
                lrs.append(grp["lr"])
                self.log("lr", lrs[0], epoch)

            logger.debug(
                f"Epoch {epoch}/{self.epochs}: Loss/train "
                f"{train_loss} Loss/valid {valid_loss}"
            )

        self.reset_logger()
        return best_model, best_optim

    def check_eval(self, eval_set: Any):
        """
        A quick check to see how the current model evaluates on the newest
        validation dataset
        """
        _, _, valid_xs, valid_ys = self._validate_args(eval_set=eval_set)

        if valid_xs is None or valid_ys is None:
            logger.warning("No eval_set provided, skipping eval")
            return None

        valid_dataloader = self.make_dataloader(valid_xs, valid_ys, self.lookback)
        criterion = torch.nn.MSELoss()
        valid_loss = 0.
        # self.nn.to(self._device)
        self.nn.eval()
        with torch.no_grad():
            for batch, batch_data in enumerate(valid_dataloader):
                x, y_target = batch_data
                x = x.float().to(self._device)  # type: ignore
                y_target = y_target.float().to(self._device)
                y_pred = self.nn(x)
                loss = criterion(y_pred, y_target)
                valid_loss += loss.item()

        valid_loss /= (float(batch+1))
        self.nn.train()
        torch.cuda.empty_cache()
        return valid_loss

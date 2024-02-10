import torch
from typing import Any
import numpy.typing as npt
from flowml.models.pytorch.base import BasePyTorchModel
import pandas as pd
from flowdapt.lib.logger import get_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from flowml.models.pytorch.utils import EarlyStopping, WindowAutoRegressiveDataset
from flowml.models.pytorch.architectures.lstm import LSTMAutoRegressive
import time

logger = get_logger(__name__)


class PyTorchLSTMAutoregressive(BasePyTorchModel):

    def fit(self,
            xs: npt.ArrayLike | pd.DataFrame,
            ys: npt.ArrayLike | pd.DataFrame,
            eval_set: Any = None):

        x, y, valid_x, valid_y = self._validate_args(xs, ys, eval_set)

        if x is None or y is None:
            logger.warning("No training data provided, skipping fit")

        dataset = self.make_dataloader(x, y, self.lookback)
        if valid_x is not None:
            valid_dataset = self.make_dataloader(valid_x, valid_y, self.lookback, batch_size=1, valid=True)
        else:
            valid_dataset = None

        input_dim = x.shape[1]
        output_dim = x.shape[1]
        # output_dim = ys.shape[1]
        self.nn = LSTMAutoRegressive(
            input_features=input_dim,
            hidden_dim=self.hidden_dim,
            output_features=output_dim,
            lookback=self.lookback
        ).to(self._device)

        self.nn, self.optimizer = self.train_loop(dataset, valid_dataset, self.nn)
        return self

    def predict(self, xs: npt.ArrayLike):
        x, _, _, _ = self._validate_args(xs)
        logger.info(f"Predicting on {x.shape[0]} samples and {x.shape[1]} features")

        self.nn.to(self._device)
        self.nn.eval()
        data = x.unsqueeze(0)
        data = data.to(self._device)
        preds, _, _ = self.get_rollout_pred(data, self.nn, hn=self.hn, cn=self.cn)
        preds = preds.cpu()
        preds = preds.squeeze()
        return preds.detach().numpy()

    def make_dataloader(self,
                        xs: torch.Tensor,
                        ys: torch.Tensor,
                        lookback: int = 10,
                        batch_size: int | None = None,
                        valid: bool = False):

        dataset = WindowAutoRegressiveDataset(
            xs, window_size=lookback, rollout=6, valid=valid)

        if batch_size is None:
            batch_size = self.batch_size

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=0,
            shuffle=self.shuffle
        )

        return dataloader

    def train_loop(self, dataloader: Any, valid_dataloader: Any, model: torch.nn.Module):

        model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, weight_decay=1e-4)
        learning_rate_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=50)

        logger.info("Starting training loop")
        min_valid_loss = 1e10
        total_batches = 0
        best_model = None
        best_optim = None

        for epoch in range(0, self.epochs):
            train_loss = 0.
            last_batch_time = time.time()
            # hn, cn = None, None
            # first = True
            for batch, batch_data in enumerate(dataloader):
                x, y_target = batch_data
                x = x.float().to(self._device)
                # if not first:
                #     hn = hn.expand(-1, x.shape[0], -1).contiguous()
                #     cn = cn.expand(-1, x.shape[0], -1).contiguous()
                y_target = y_target.float().to(self._device)
                loss, _, _, _ = self.get_rollout_loss(x, y_target, model, criterion)
                # Backprogation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_batches += 1
                train_loss += loss.item()
                self.log("train_loss", loss.item() /
                                     (self.batch_size * 6), total_batches)
                total_batches += 1

                # # normalize the hidden states across the batch
                # hn = torch.mean(hn.detach(), dim=1).unsqueeze(1)
                # cn = torch.mean(cn.detach(), dim=1).unsqueeze(1)
                # first = False

            samples_per_second = self.batch_size / (time.time() - last_batch_time)
            self.log("samples_per_second", samples_per_second, epoch)

            valid_loss = 0.
            if valid_dataloader is not None:
                # get loss on valid_dataset
                model.eval()
                # hn_v = None  # hn.detach()
                # cn_v = None  # cn.detach()
                with torch.no_grad():
                    for batch, batch_data in enumerate(valid_dataloader):
                        x, y_target = batch_data
                        x = x.float().to(self._device)
                        y_target = y_target.float().to(self._device)
                        loss, _, _, _ = self.get_rollout_loss(x, y_target, model, criterion)
                        # , hn=hn_v, cn=cn_v)
                        valid_loss += loss.item()

                valid_loss /= (float(batch+1) * self.batch_size * 6)  # 6 is the rollout
                model.train()
                learning_rate_scheduler.step(valid_loss)

                # save best model for later return
                if valid_loss < min_valid_loss:
                    logger.debug(f"found best model with loss {valid_loss}")
                    best_model = model
                    best_optim = optimizer
                    min_valid_loss = valid_loss
                    self.hn = None  # hn_v.detach()
                    self.cn = None  # cn_v.detach()
                    self.best_loss = float(min_valid_loss)

                if early_stopping(valid_loss) and self.early_stopping:
                    logger.debug("Model is not improving, Early stopping activated")
                    break

                self.log("valid_loss", valid_loss, step=epoch)

            lrs = []
            for grp in optimizer.param_groups:
                lrs.append(grp["lr"])
                self.log("lr", lrs[0], epoch)

            logger.debug(f"Epoch {epoch}/{self.epochs}: Loss/train "
                         f"{train_loss} Loss/valid {valid_loss}")

        self.reset_logger()
        if not best_model:
            best_model = model
            best_optim = optimizer
        return best_model, best_optim

    def get_rollout_loss(self, x, y_target, model, criterion, hn=None, cn=None):
        """
        Given the X and y_target, compute predictions and loss for each step
        along the desired rollout. Collect full loss during rollout and return
        loss to the optimizer.
        """
        logger.debug(f"x, y shape {x.shape}, {y_target.shape}")

        batch_size = x.shape[0]
        rollout = y_target.shape[1]
        # loop through the columns of y_target at intervals of self.lookback
        # for t in np.arange(0, y_target.shape[1], self.output_dim):
        for t in range(0, rollout):
            y = y_target[:, t, :]

            im, hn, cn = model(x, hn, cn)
            if t == 0:
                loss = criterion(
                    im.reshape(batch_size, -1), y.reshape(batch_size, -1)
                )
                pred = im
            else:
                loss += criterion(
                    im.reshape(batch_size, -1), y.reshape(batch_size, -1)
                )
                pred = torch.cat((pred, im), dim=1)
            x = torch.cat((x[:, 1:, :], im), dim=1)

        return loss, pred, hn, cn

    def get_rollout_pred(self, x, model, rollout=6, hn=None, cn=None):
        """
        Given data to inference, x, compute the preidictions for a given
        rollout. Return the predictions
        """
        for t in range(0, rollout):
            im, hn, cn = model(x, hn, cn)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=1)
            x = torch.cat((x[:, 1:, :], im), dim=1)

        return pred, hn, cn

from torch.utils.tensorboard import SummaryWriter
from typing import Any
import xgboost as xgb


class TensorboardLogger:
    def __init__(self, logdir: str = "tensorboard", id: str = "unique-id"):
        self.writer: SummaryWriter = SummaryWriter(f"{logdir}/model_{id}")

    def add_scalar(self, tag: str, scalar_value: Any, step: int):
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class TensorBoardCallback(xgb.callback.TrainingCallback):

    def __init__(self, logdir: str = "tensorboard", id: str = "unique-id"):
        self.writer: SummaryWriter = SummaryWriter(logdir)

    def after_iteration(
        self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    # FIXME: ensure we can get the MSE for XGBoost without squaring this
                    self.writer.add_scalar("train_loss", score, epoch)
                else:
                    self.writer.add_scalar("valid_loss", score, epoch)

        return False

    def after_training(self, model):
        self.writer.flush()
        self.writer.close()

        return model
